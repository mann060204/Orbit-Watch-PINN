"""Train and evaluate an actual PINN; save every artifact in a dedicated run folder."""
from __future__ import annotations

import argparse
from datetime import timedelta
import hashlib
import json
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import torch

from collision_probability import apply_covariances
from fetch_celestrak import parse_epoch, utc_now, utc_text, write_json
from orbit_engine import AnalysisConfig, prepare_catalog, propagate, screen
from pinn_device import inspect_device
from pinn_evaluation import event_comparison, pair_classification, trajectory_metrics
from pinn_model import LENGTH_KM, SPEED_KM_S, normalize_states, rollout
from pinn_screening import screen_pinn
from run_analysis import ROOT, read_snapshot, write_csv
from train_pinn import split_objects, train

SOURCE_FILES=["pinn_model.py","pinn_device.py","train_pinn.py","run_pinn.py","pinn_screening.py",
              "pinn_evaluation.py","collision_probability.py","orbit_engine.py"]


def run_pinn(steps=6000,hours=1,segment_seconds=60,threshold_km=5,start=None,seed=2026,
             covariance_path=None,progress=lambda *args:None):
    if not 100<=steps<=30000:raise ValueError("Training steps must be between 100 and 30000")
    if not 0.1<=hours<=6:raise ValueError("This initial PINN supports experiments from 0.1 to 6 hours")
    if not 10<=segment_seconds<=60:raise ValueError("Neural integration segments must be 10 to 60 seconds")
    if not 0.01<=threshold_km<=50:raise ValueError("Screening threshold must be 0.01 to 50 km")
    began=time.perf_counter()
    hardware=inspect_device()
    start=start or utc_now()
    snapshot,input_manifest,records=read_snapshot()
    folder=ROOT/"pinn_results"/utc_now().strftime("%Y%m%dT%H%M%S_%fZ")
    folder.mkdir(parents=True,exist_ok=False)
    logs=[]
    def status(percent,message,extra=None):
        value={"status":"running","run_id":folder.name,"progress":percent,"message":message,
               "updated_at_utc":utc_text(utc_now())}
        if extra:value["training"]=extra
        write_json(folder/"status.json",value)
        write_json(ROOT/"pinn_results"/"active.json",{"run_id":folder.name})
        logs.append(f"{value['updated_at_utc']} | {percent:.2f}% | {message}")
        progress(percent,message)
    try:
        status(1,"Checking CPU capability and preparing actual catalog initial states")
        write_json(folder/"hardware.json",hardware)
        config={"steps":steps,"horizon_hours":hours,"segment_seconds":segment_seconds,"threshold_km":threshold_km,
                "max_epoch_age_hours":72,"seed":seed,"start_utc":utc_text(start),"dtype":"float64",
                "architecture":"Structured Taylor trial solution with learned radial/velocity/axis corrections",
                "physics":"central gravity + J2","future_sgp4_training_targets":0}
        write_json(folder/"config.json",config)
        shutil.copytree(snapshot,folder/"input_snapshot")
        satellites,rows,initialization_errors=prepare_catalog(records)
        initial_errors,initial_r,initial_v=propagate(satellites,start,[0.0])
        selected=[];exclusions=list(initialization_errors)
        for index,row in enumerate(rows):
            age=(start-parse_epoch(row["EPOCH"])).total_seconds()/3600
            reason="Epoch older than 72 h" if abs(age)>72 else "Invalid initial SGP4 state" if initial_errors[index,0] or not np.isfinite(initial_r[index,0]).all() else None
            if reason:exclusions.append({"norad_id":int(row["NORAD_CAT_ID"]),"reason":reason,"epoch_age_hours":age})
            else:selected.append(index)
        if len(selected)<30:raise ValueError("Fewer than 30 fresh valid objects available for grouped train/validation/test splits")
        rows=[rows[index] for index in selected];satellites=[satellites[index] for index in selected]
        initial_states=normalize_states(initial_r[selected,0],initial_v[selected,0])
        ids=np.array([int(row["NORAD_CAT_ID"]) for row in rows])
        partitions=split_objects(rows,seed)
        write_json(folder/"excluded_objects.json",exclusions)
        write_json(folder/"data_splits.json",{name:ids[indices].tolist() for name,indices in partitions.items()})
        np.savez_compressed(folder/"initial_conditions.npz",norad_ids=ids,normalized_state=initial_states,
                            epoch_utc=np.array([row["EPOCH"] for row in rows]),start_utc=np.array(utc_text(start)))
        history_stream=folder/"training_history.jsonl"
        def training_progress(fraction,message,row):
            with history_stream.open("a",encoding="utf-8") as stream:stream.write(json.dumps(row)+"\n")
            status(5+65*fraction,message,row)
        status(4,f"Training on {len(partitions['train']):,} catalog objects; {len(partitions['test']):,} held-out test objects")
        model,untrained,history,proof=train(initial_states,partitions,hardware,steps,segment_seconds,seed,
                                             training_progress,folder/"model.pt")
        write_csv(folder/"training_history.csv",history)
        write_json(folder/"training_proof.json",proof)
        status(72,"Rolling out trained neural predictions; evaluating SGP4 references separately")
        seconds=np.append(np.arange(0,hours*3600,segment_seconds,dtype=float),hours*3600)
        prediction=rollout(model,initial_states,seconds,hardware["selected_device"])
        ablation=rollout(untrained,initial_states,seconds,"cpu")
        if not np.isfinite(prediction).all():raise ValueError("Neural rollout diverged; predictions contain nonfinite values")
        # Only now, after training is finished, are future reference trajectories generated.
        errors,reference_r,reference_v=propagate(satellites,start,seconds)
        if (errors!=0).any() or not np.isfinite(reference_r).all():
            raise ValueError("Some future SGP4 reference states failed; a valid comparison cannot be produced")
        np.savez_compressed(folder/"pinn_trajectories.npz",norad_ids=ids,seconds_from_start=seconds,
                            start_utc=np.array(utc_text(start)),position_teme_km=prediction[:,:,:3]*LENGTH_KM,
                            velocity_teme_km_s=prediction[:,:,3:]*SPEED_KM_S)
        np.savez_compressed(folder/"sgp4_reference_trajectories.npz",norad_ids=ids,seconds_from_start=seconds,
                            position_teme_km=reference_r,velocity_teme_km_s=reference_v,sgp4_errors=errors)
        errors_metrics,position_errors,velocity_errors=trajectory_metrics(prediction,reference_r,reference_v,partitions)
        ablation_metrics,_,_=trajectory_metrics(ablation,reference_r,reference_v,partitions)
        np.savez_compressed(folder/"reference_errors.npz",norad_ids=ids,seconds_from_start=seconds,
                            position_vector_error_km=position_errors,velocity_vector_error_km_s=velocity_errors)
        object_metrics=[]
        split_names={int(index):name for name,indices in partitions.items() for index in indices}
        for index,row in enumerate(rows):
            object_metrics.append({"norad_id":int(ids[index]),"name":row.get("OBJECT_NAME",""),"split":split_names[index],
                                   "epoch_utc":row["EPOCH"],"position_rmse_km":float(np.sqrt(np.mean(position_errors[index,1:]**2))),
                                   "final_position_error_km":float(position_errors[index,-1]),
                                   "velocity_rmse_km_s":float(np.sqrt(np.mean(velocity_errors[index,1:]**2)))})
        write_csv(folder/"per_object_evaluation.csv",object_metrics)
        status(78,"Building the independent SGP4 screening reference")
        screening_config=AnalysisConfig(hours,segment_seconds,threshold_km,72,False)
        reference_events,audit,ref_excluded,refinement_errors,reference_stats=screen(
            satellites,rows,reference_r,errors,start,seconds,screening_config)
        if refinement_errors:raise ValueError("SGP4 reference refinement failed; comparison metrics would be incomplete")
        shared=reference_stats["shared_orbit_pairs"]
        write_json(folder/"sgp4_reference_events.json",reference_events)
        write_csv(folder/"shared_orbit_pairs.csv",shared,["object1_id","object2_id","object1_name","object2_name","reason"])
        status(83,"Screening neural trajectories and refining times with the neural model")
        pinn_events,pinn_audit,pinn_stats=screen_pinn(model,prediction,rows,start,seconds,threshold_km,shared,
                                                     hardware["selected_device"],lambda message:status(86,message))
        covariances=json.loads(Path(covariance_path).read_text(encoding="utf-8")) if covariance_path else None
        if covariances is not None:write_json(folder/"supplied_covariances.json",covariances)
        probability_status=apply_covariances(pinn_events,covariances)
        write_json(folder/"pinn_conjunctions.json",pinn_events)
        write_csv(folder/"pinn_conjunctions.csv",pinn_events,["event_id","object1_id","object2_id","object1_name","object2_name",
                  "tca_utc","miss_distance_km","relative_speed_km_s","collision_probability","probability_status"])
        write_csv(folder/"pinn_candidate_audit.csv",pinn_audit,["a","b","interval","offset_seconds","miss_distance_km","relative_speed_km_s"])
        classification,pair_rows=pair_classification(pinn_events,reference_events,ids,shared)
        heldout_classification,_=pair_classification(pinn_events,reference_events,ids[partitions["test"]],shared)
        write_csv(folder/"pair_classification_vs_sgp4.csv",pair_rows,["object1_id","object2_id","pinn_positive","sgp4_reference_positive","outcome"])
        write_json(folder/"pair_universe.json",{"norad_ids":ids.tolist(),"excluded_identical_orbit_pairs":shared,
                   "definition":"All unordered distinct pairs of these IDs except the listed shared pairs. Unlisted pairs in pair_classification_vs_sgp4.csv are true negatives relative to SGP4, not observed collision negatives."})
        comparison=event_comparison(pinn_events,reference_events)
        write_csv(folder/"event_comparison_vs_sgp4.csv",comparison,["event_id","object1_id","object2_id","pinn_miss_distance_km",
                  "pinn_tca_utc","sgp4_miss_distance_km","sgp4_tca_utc","miss_distance_difference_km","tca_difference_seconds"])
        metrics={"observational_collision_metrics":{"status":"not_evaluable","accuracy":None,"precision":None,"recall":None,
                 "f1_score":None,"roc_auc":None,"pr_auc":None,"brier_score":None,
                 "reason":"No observed collision outcomes or independent measured trajectories provided."},
                 "sgp4_reference_pair_classification":classification,"heldout_object_pair_classification_vs_sgp4":heldout_classification,
                 "trajectory_agreement_vs_sgp4":errors_metrics,"untrained_taylor_ablation_vs_sgp4":ablation_metrics,
                 "training":proof,"collision_probability":probability_status}
        write_json(folder/"metrics.json",metrics)
        flat=[]
        def flatten(value,prefix=""):
            for key,item in value.items():
                name=f"{prefix}.{key}" if prefix else key
                if isinstance(item,dict):flatten(item,name)
                else:flat.append({"metric":name,"value":item,"status":"unavailable" if item is None else "recorded"})
        flatten(metrics);write_csv(folder/"metrics.csv",flat)
        matched=[row for row in comparison if row["sgp4_miss_distance_km"] is not None]
        summary={"schema_version":1,"run_id":folder.name,"config":config,"hardware":hardware,
                 "input_catalog_objects":len(records),"modeled_objects":len(rows),"excluded_objects":len(exclusions),
                 "split_counts":{name:len(indices) for name,indices in partitions.items()},
                 "window_start_utc":utc_text(start),"window_end_utc":utc_text(start+timedelta(hours=hours)),
                 "input_manifest":input_manifest,"training":proof,"metrics":metrics,
                 "pinn_close_approach_events":len(pinn_events),"sgp4_reference_events":len(reference_events),
                 "closest_pinn_approach_km":pinn_events[0]["miss_distance_km"] if pinn_events else None,
                 "pinn_screening":pinn_stats,"shared_orbit_pairs":len(shared),
                 "matched_event_comparison":{"count":len(matched),
                    "mean_absolute_miss_distance_difference_km":float(np.mean([abs(row['miss_distance_difference_km']) for row in matched])) if matched else None,
                    "mean_absolute_tca_difference_seconds":float(np.mean([abs(row['tca_difference_seconds']) for row in matched])) if matched else None},
                 "model_limitations":["Central gravity and J2 only; drag, J3/J4, third bodies and maneuvers are omitted.",
                    "TEME is treated as quasi-inertial for this short window; Earth orientation transformations are not modeled.",
                    "A learned 60-second (or configured shorter) flow map is iterated; errors can accumulate over longer forecasts.",
                    "Physics collocation is not measured training data. CelesTrak GP initial conditions are orbital estimates.",
                    "SGP4 is a model reference, not observational truth. Agreement scores are not real collision accuracy.",
                    "A structured analytical Taylor trial solution supplies lower-order dynamics; the trained network learns corrections. The untrained ablation is reported.",
                    "Collision probability requires supplied covariance and object radii; no Monte Carlo samples or assumed uncertainties are used.",
                    "Conjunction screening uses conservative assumed acceleration padding and local minimization; it is not a proof of complete detection."],
                 "source_sha256":{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCE_FILES}}
        status(94,"Saving evaluation plots and the complete neural-network report")
        from pinn_report import make_report
        make_report(folder,summary,history,seconds,position_errors,partitions,pinn_events)
        summary["total_runtime_seconds"]=time.perf_counter()-began
        write_json(folder/"summary.json",summary)
        status(100,f"Completed actual PINN training and evaluation: {len(pinn_events)} neural close approaches")
        write_json(folder/"status.json",{"status":"complete","run_id":folder.name,"progress":100,
                                         "message":"Training and evaluation complete","updated_at_utc":utc_text(utc_now())})
        write_json(ROOT/"pinn_results"/"latest.json",{"run_id":folder.name})
        return folder,summary
    except Exception as exc:
        write_json(folder/"status.json",{"status":"failed","run_id":folder.name,"error":str(exc)})
        (folder/"error.txt").write_text(traceback.format_exc(),encoding="utf-8")
        raise
    finally:
        (folder/"run.log").write_text("\n".join(logs)+"\n",encoding="utf-8")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps",type=int,default=6000)
    parser.add_argument("--hours",type=float,default=1)
    parser.add_argument("--segment-seconds",type=float,default=60)
    parser.add_argument("--threshold-km",type=float,default=5)
    parser.add_argument("--seed",type=int,default=2026)
    parser.add_argument("--start")
    parser.add_argument("--covariances",type=Path)
    args=parser.parse_args()
    folder,_=run_pinn(args.steps,args.hours,args.segment_seconds,args.threshold_km,
                      parse_epoch(args.start) if args.start else None,args.seed,args.covariances,
                      lambda value,message:print(f"{value:5.1f}% {message}",flush=True))
    print(f"PINN results: {folder}")
