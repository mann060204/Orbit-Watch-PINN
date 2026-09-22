"""Train and export SGP4, standalone PINN, and SGP4+neural correction locally."""
from __future__ import annotations

import argparse
from datetime import timedelta
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import torch
from sgp4.exporter import export_tle

from comparison_evaluation import flatten_metrics, physics_diagnostics
from evaluation import CLASSIFICATION_NAMES
from fetch_celestrak import parse_epoch, utc_now, utc_text, validate_records, write_json
from hybrid_model import load_corrector, predict_corrected, train_corrector
from orbit_engine import make_satellite, propagate, reference_validation
from orbit_truth import orbit_error_metrics, read_esa_orbit, reference_to_teme, select_native_samples
from pinn_device import inspect_device
from pinn_model import LENGTH_KM, SPEED_KM_S, OrbitalPINN, normalize_states, rollout
from run_analysis import write_csv
from train_pinn import train

ROOT = Path(__file__).resolve().parent
MODEL_FOLDERS = {"SGP4":"sgp4", "PINN":"pinn", "COMBINED":"combined"}
SPLIT_BOUNDS = {"train":(0, 90), "validation":(90, 120), "test":(120, 180)}
SOURCE_FILES = ["run_three_models.py", "hybrid_model.py", "three_model_report.py", "verify_three_model_run.py",
                "publish_model_results.py",
                "pinn_model.py", "pinn_device.py", "train_pinn.py", "orbit_truth.py", "orbit_engine.py",
                "comparison_evaluation.py", "fetch_celestrak.py", "run_analysis.py", "evaluation.py",
                "requirements-lock.txt", "requirements-pinn.txt", "requirements-truth.txt", "RunThreeModels.bat"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def split_metrics(prediction, truth_r, truth_v):
    """The preceding boundary is diagnostic only; each scored timestamp occurs once."""
    return {name:orbit_error_metrics(prediction["r"][a:b+1], prediction["v"][a:b+1],
                                     truth_r[a:b+1], truth_v[a:b+1])[0]
            for name, (a, b) in SPLIT_BOUNDS.items()}


def state_rows(seconds, start, r, v):
    return [{"utc":utc_text(start+timedelta(seconds=float(second))), "seconds_from_start":float(second),
             **{axis+"_km":float(r[i,j]) for j,axis in enumerate("xyz")},
             **{"v"+axis+"_km_s":float(v[i,j]) for j,axis in enumerate("xyz")}}
            for i,second in enumerate(seconds)]


def prepare_training_catalog(base_run, folder, benchmark_id):
    """Verify and rederive earlier catalog inputs; never reuse old trained weights."""
    config = json.loads((base_run/"config.json").read_text())
    checksums = json.loads((base_run/"file_checksums.json").read_text())
    for name in ("config.json", "eligible_objects.json", "data_splits.json", "initial_conditions.npz"):
        if sha(base_run/name) != checksums[name]:
            raise ValueError(f"Original training catalog checksum mismatch: {name}")
        shutil.copy2(base_run/name, folder/name)
    source = (base_run/config["input_snapshot"]).resolve()
    if not source.is_relative_to(base_run.resolve()):
        raise ValueError("Training snapshot must be inside its original run")
    for path in source.rglob("*"):
        if path.is_file() and sha(path) != checksums[path.relative_to(base_run).as_posix()]:
            raise ValueError(f"Training source checksum mismatch: {path.name}")
    shutil.copytree(source, folder/"input_snapshot")
    rows = validate_records(json.loads((folder/"eligible_objects.json").read_text()))
    ids = np.array([int(row["NORAD_CAT_ID"]) for row in rows])
    if benchmark_id in ids:
        raise ValueError("Independent benchmark object appears in PINN training catalog")
    with np.load(folder/"initial_conditions.npz", allow_pickle=False) as saved:
        initial, saved_ids = saved["normalized_state"], saved["norad_ids"]
    codes, r0, v0 = propagate([make_satellite(row) for row in rows], parse_epoch(config["start_utc"]), [0.0])
    if np.any(codes) or not np.array_equal(ids, saved_ids) or not np.allclose(initial, normalize_states(r0[:,0], v0[:,0]), rtol=0, atol=1e-12):
        raise ValueError("Saved catalog initial conditions do not replay from actual CelesTrak GP")
    raw_splits = json.loads((folder/"data_splits.json").read_text())
    index = {int(norad_id):i for i,norad_id in enumerate(ids)}
    partitions = {name:np.array([index[int(n)] for n in raw_splits[name]], dtype=int)
                  for name in ("train", "validation", "test")}
    flat = np.concatenate(list(partitions.values()))
    if len(flat) != len(ids) or len(np.unique(flat)) != len(ids) or any(not len(x) for x in partitions.values()):
        raise ValueError("PINN catalog splits must be disjoint and cover every object")
    return initial, partitions, {"objects":len(ids), "start_utc":config["start_utc"],
                                 "split_counts":{k:len(v) for k,v in partitions.items()},
                                 "benchmark_object_excluded":True, "rederived_initial_states_verified":True}


def run(pinn_steps=6000, correction_steps=3000, seed=2026, input_dir=None, training_run=None):
    if type(pinn_steps) is not int or not 100 <= pinn_steps <= 30000:
        raise ValueError("Use 100 to 30000 PINN steps")
    if type(correction_steps) is not int or not 100 <= correction_steps <= 30000:
        raise ValueError("Use 100 to 30000 correction steps")
    input_dir = Path(input_dir or ROOT/"reference_inputs/sentinel_benchmark").resolve()
    training_run = Path(training_run or ROOT/"comparison_results/20260911T100012_038157Z").resolve()
    output = ROOT/"model_results"
    folder = output/utc_now().strftime("%Y%m%dT%H%M%S_%fZ")
    folder.mkdir(parents=True, exist_ok=False)
    for name in ("sgp4", "pinn", "combined", "comparison", "reference", "inputs/benchmark", "inputs/training_catalog", "source_code"):
        (folder/name).mkdir(parents=True)
    began = time.perf_counter()
    def status(message):
        print(message, flush=True)
        write_json(folder/"status.json", {"status":"running", "message":message, "run_id":folder.name})
        with (folder/"run.log").open("a", encoding="utf-8") as stream:
            stream.write(f"{utc_text(utc_now())} | {message}\n")
    try:
        config = {"pinn_steps":pinn_steps, "correction_steps":correction_steps, "seed":seed,
                  "step_seconds":60, "horizon_seconds":10800, "train_end_seconds":5400,
                  "validation_end_seconds":7200, "test_end_seconds":10800,
                  "primary_test_indices":list(range(121,181)), "website_publication":False,
                  "training_run":str(training_run), "reference_inputs":str(input_dir),
                  "method_locked_before_evaluation":True,
                  "information_budget":"Hybrid uses earlier ESA calibration and validation states; standalone PINN sees no ESA states."}
        write_json(folder/"config.json", config)
        status("Verifying saved real CelesTrak, ESA and Earth-orientation source files")
        manifest = json.loads((input_dir/"source_manifest.json").read_text())
        for name, expected in manifest["files_sha256"].items():
            if Path(name).name != name or sha(input_dir/name) != expected:
                raise ValueError(f"Reference source checksum failed: {name}")
            shutil.copy2(input_dir/name, folder/"inputs/benchmark"/name)
        for name in ("source_manifest.json", "finals2000A.all", "finals2000A.source.json"):
            shutil.copy2(input_dir/name, folder/"inputs/benchmark"/name)
        eop_note = json.loads((folder/"inputs/benchmark/finals2000A.source.json").read_text())
        if eop_note["sha256"] != sha(folder/"inputs/benchmark/finals2000A.all"):
            raise ValueError("Earth orientation checksum failed")
        records = validate_records(json.loads((folder/"inputs/benchmark/initial_gp.json").read_text()))
        if len(records) != 1 or int(records[0]["NORAD_CAT_ID"]) != manifest["norad_cat_id"]:
            raise ValueError("Expected one reference-matching CelesTrak GP record")
        record = records[0]
        reference_name = manifest["reference_source"]["eof_file"]
        if Path(reference_name).name != reference_name:
            raise ValueError("Reference product must be a basename")
        reference = read_esa_orbit(folder/"inputs/benchmark"/reference_name)
        if reference["mission"].replace("-", "").upper() != record["OBJECT_NAME"].replace("-", "").upper():
            raise ValueError("ESA and GP objects differ")
        indices, seconds = select_native_samples(reference, parse_epoch(record["EPOCH"]), 3, 60)
        start = reference["times"][indices[0]]
        hardware = inspect_device()
        torch.set_num_threads(hardware["torch_threads"])
        torch.use_deterministic_algorithms(True)
        device = hardware["selected_device"]
        write_json(folder/"hardware.json", hardware)
        initial_catalog, partitions, catalog_proof = prepare_training_catalog(training_run, folder/"inputs/training_catalog", int(record["NORAD_CAT_ID"]))
        if parse_epoch(catalog_proof["start_utc"]) >= start:
            raise ValueError("PINN training catalog must predate the independent test arc")
        status(f"Training standalone PINN from scratch on {catalog_proof['objects']:,} verified CelesTrak initial states ({device})")
        def progress_pinn(fraction, message, row):
            status(message)
            with (folder/"pinn/training_history.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row)+"\n")
        model, untrained, pinn_history, pinn_proof = train(initial_catalog, partitions, hardware, pinn_steps, 60, seed,
                                                          progress_pinn, folder/"pinn/model.pt")
        torch.save({"model_state_dict":untrained.state_dict(), "width":untrained.width, "depth":untrained.depth}, folder/"pinn/untrained_model.pt")
        pinn_proof["catalog"] = catalog_proof
        write_json(folder/"pinn/training_proof.json", pinn_proof)
        write_csv(folder/"pinn/training_history.csv", pinn_history)
        status("Computing independent SGP4 and PINN trajectories from one shared GP-derived initial state")
        satellite = make_satellite(record)
        codes, sr, sv = propagate([satellite], start, seconds)
        if np.any(codes) or not np.isfinite(sr).all() or not np.isfinite(sv).all():
            raise ValueError("Invalid SGP4 forecast state")
        sr, sv = sr[0], sv[0]
        initial = normalize_states(sr[:1], sv[:1])
        def neural_prediction():
            normalized = rollout(model, initial, seconds, device)[0]
            return {"r":normalized[:, :3]*LENGTH_KM, "v":normalized[:, 3:]*SPEED_KM_S}
        predictions = {"SGP4":{"r":sr, "v":sv}, "PINN":neural_prediction()}
        tle = export_tle(satellite)
        (folder/"inputs/derived_initial_tle.txt").write_text(record["OBJECT_NAME"]+"\n"+"\n".join(tle)+"\n", encoding="utf-8")
        np.savez_compressed(folder/"inputs/common_initial_state.npz", normalized_state=initial, position_teme_km=sr[0], velocity_teme_km_s=sv[0], start_utc=np.array(utc_text(start)))
        status("Preparing only earlier ESA samples for correction training and validation")
        # Held-out reference states are neither transformed nor scored until weights freeze.
        early_r, early_v, frame_info, early_eop = reference_to_teme(reference, indices[:121], folder/"inputs/benchmark/finals2000A.all")
        def calibration_slice(a, b):
            return {"seconds":seconds[a:b].copy(), "sgp4_r":sr[a:b].copy(), "sgp4_v":sv[a:b].copy(),
                    "reference_r":early_r[a:b].copy(), "reference_v":early_v[a:b].copy()}
        training, validation = calibration_slice(1, 91), calibration_slice(91, 121)
        period = 86400 / float(record["MEAN_MOTION"])
        def progress_correction(fraction, message, row):
            status(message)
            with (folder/"combined/training_history.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row)+"\n")
        corrector, correction_config, correction_history, correction_proof = train_corrector(training, validation, hardware, period,
                         correction_steps, seed, folder/"combined/model.pt", progress_correction)
        write_json(folder/"combined/training_proof.json", correction_proof)
        write_csv(folder/"combined/training_history.csv", correction_history)
        for name, values in (("train", training), ("validation", validation)):
            np.savez_compressed(folder/"combined"/(name+"_calibration_samples.npz"), **values)
        frozen_hash = sha(folder/"combined/model.pt")
        status("Correction weights frozen; generating the combined forecast and opening the held-out reference hour")
        predictions["COMBINED"] = predict_corrected(corrector, correction_config, seconds, sr, sv, device)
        test_r, test_v, test_frame, test_eop = reference_to_teme(reference, indices[121:], folder/"inputs/benchmark/finals2000A.all")
        truth_r, truth_v = np.vstack((early_r, test_r)), np.vstack((early_v, test_v))
        common = {"seconds_from_start":seconds, "utc":np.array([utc_text(reference["times"][i]) for i in indices]), "norad_id":np.array(record["NORAD_CAT_ID"])}
        np.savez_compressed(folder/"reference/orbit.npz", **common, position_teme_km=truth_r, velocity_teme_km_s=truth_v,
                            position_itrs_km=reference["r_itrs_km"][indices], velocity_itrs_km_s=reference["v_itrs_km_s"][indices], original_osv_indices=indices)
        write_csv(folder/"reference/states.csv", state_rows(seconds, start, truth_r, truth_v))
        write_json(folder/"reference/frame_transformation.json", {"calibration":frame_info, "test":test_frame, "eop_sha256":sha(folder/"inputs/benchmark/finals2000A.all")})
        write_csv(folder/"reference/earth_orientation.csv", early_eop+test_eop)
        errors, metrics, flat_metrics, timing_rows = {}, {}, [], []
        status("Measuring propagation runtimes and computing held-out RMSE, MAE and RTN errors")
        def evaluate(name):
            if name == "SGP4":
                return propagate([satellite], start, seconds)
            if name == "PINN":
                return neural_prediction()
            _, r, v = propagate([satellite], start, seconds)
            return predict_corrected(corrector, correction_config, seconds, r[0], v[0], device)
        for name in MODEL_FOLDERS:
            evaluate(name)
        for repeat in range(5):
            names = list(MODEL_FOLDERS)
            names = names[repeat%3:] + names[:repeat%3]
            for order, name in enumerate(names):
                if device == "cuda": torch.cuda.synchronize()
                began_prediction = time.perf_counter()
                evaluate(name)
                if device == "cuda": torch.cuda.synchronize()
                timing_rows.append({"model":name, "repeat":repeat+1, "order":order+1, "seconds":time.perf_counter()-began_prediction})
        timing = {name:float(np.median([r["seconds"] for r in timing_rows if r["model"]==name])) for name in MODEL_FOLDERS}
        write_csv(folder/"comparison/propagation_benchmarks.csv", timing_rows)
        for name, values in predictions.items():
            destination = folder/MODEL_FOLDERS[name]
            metrics[name] = split_metrics(values, truth_r, truth_v)
            _, errors[name] = orbit_error_metrics(values["r"], values["v"], truth_r, truth_v)
            np.savez_compressed(destination/"predictions.npz", **common, position_teme_km=values["r"], velocity_teme_km_s=values["v"])
            np.savez_compressed(destination/"errors.npz", **common, **errors[name])
            write_csv(destination/"predictions.csv", state_rows(seconds, start, values["r"], values["v"]))
            error_rows = []
            for i, second in enumerate(seconds):
                split = "initial" if i == 0 else "train" if i <= 90 else "validation" if i <= 120 else "test"
                row = {"utc":str(common["utc"][i]), "seconds_from_start":float(second), "split":split}
                for field, values_error in errors[name].items():
                    if values_error.ndim == 1: row[field] = float(values_error[i])
                    else:
                        axes = ("radial", "transverse", "normal") if "rtn" in field else ("x", "y", "z")
                        row.update({field+"_"+axis:float(values_error[i,j]) for j,axis in enumerate(axes)})
                error_rows.append(row)
            write_csv(destination/"errors_by_time.csv", error_rows)
            write_json(destination/"metrics.json", metrics[name])
            model_flat = [{"model":name, **row} for row in flatten_metrics(metrics[name])]
            flat_metrics.extend(model_flat)
            write_csv(destination/"metrics.csv", model_flat)
        write_json(folder/"sgp4/training_proof.json", {"training_applicable":False, "training_seconds":0.0,
                   "reason":"SGP4 is a deterministic propagator of supplied GP mean elements; it is not a neural network."})
        np.savez_compressed(folder/"sgp4/error_codes.npz", codes=codes)
        write_json(folder/"comparison/metrics.json", metrics)
        write_csv(folder/"comparison/all_metrics.csv", flat_metrics)
        test_rows = [{"model":name, **{k:v for k,v in metrics[name]["test"].items() if isinstance(v, (int,float))},
                      "propagation_seconds_median":timing[name]} for name in MODEL_FOLDERS]
        write_csv(folder/"comparison/test_metrics.csv", test_rows)
        diagnostics = {name:physics_diagnostics(p["r"][None], p["v"][None], seconds) for name,p in predictions.items()}
        write_json(folder/"comparison/physics_diagnostics.json", diagnostics)
        pairwise = {a+"_vs_"+b:orbit_error_metrics(predictions[a]["r"][120:], predictions[a]["v"][120:],
                       predictions[b]["r"][120:], predictions[b]["v"][120:])[0]
                    for a,b in (("PINN","SGP4"), ("COMBINED","SGP4"), ("COMBINED","PINN"))}
        write_json(folder/"comparison/model_agreement_not_ground_truth.json", pairwise)
        unavailable = {name:None for name in CLASSIFICATION_NAMES}
        unavailable.update(roc_auc=None, pr_auc=None, brier_score=None, collision_probability=None,
                           close_approach_events=None, reason="One independent satellite orbit: no second reference trajectory, labeled conjunction outcomes, covariance or physical radii. Orbital regression errors are measurable; collision detection accuracy/probability are not.")
        write_json(folder/"comparison/unavailable_collision_metrics.json", {name:unavailable for name in MODEL_FOLDERS})
        for name in MODEL_FOLDERS:
            write_json(folder/MODEL_FOLDERS[name]/"unavailable_collision_metrics.json", unavailable)
        checks = {"same_initial_r_and_v":all(np.allclose(p["r"][0], sr[0], rtol=0, atol=1e-10) and np.allclose(p["v"][0], sv[0], rtol=0, atol=1e-12) for p in predictions.values()),
                  "all_states_finite":all(np.isfinite(p["r"]).all() and np.isfinite(p["v"]).all() for p in predictions.values()),
                  "exactly_60_heldout_samples_each":all(m["test"]["future_samples"]==60 for m in metrics.values()),
                  "correction_frozen_before_test":sha(folder/"combined/model.pt")==frozen_hash,
                  "no_test_samples_in_training":correction_proof["test_samples_received"]==0 and correction_proof["validation_seconds_max"]==7200,
                  "pinn_no_esa_training_labels":pinn_proof["measured_future_training_samples"]==0,
                  "benchmark_absent_from_pinn_catalog":catalog_proof["benchmark_object_excluded"],
                  "all_reference_quality_nominal":bool(np.all(reference["quality"][indices]=="NOMINAL")),
                  "sgp4_reference_implementation_passed":reference_validation()["passed"],
                  "rtn_norms_preserved":all(np.allclose(np.linalg.norm(e["position_rtn_km"],axis=1), e["position_norm_km"], atol=1e-10) for e in errors.values())}
        if not all(checks.values()): raise ValueError(f"Validation failed: {checks}")
        write_json(folder/"comparison/verification.json", checks)
        meta = {"run_id":folder.name, "object_name":record["OBJECT_NAME"], "norad_id":int(record["NORAD_CAT_ID"]),
                "start_utc":utc_text(start), "end_utc":utc_text(start+timedelta(hours=3)), "reference_label":"ESA GNSS restituted orbit (AUX_RESORB)",
                "reference_is_error_free":False, "reference_is_final_precise":False, "retrospective":True,
                "training_proofs":{"PINN":pinn_proof, "COMBINED":correction_proof}, "hardware":hardware,
                "prediction_timing_seconds":timing, "split":config, "metrics":metrics,
                "source_manifest":manifest, "website_publication":False,
                "timing_note":"Median of five warmed measurements, rotated order; combined includes SGP4 propagation. Training and file I/O excluded.",
                "package_versions":{name:version(name) for name in ("torch", "numpy", "scipy", "sgp4", "astropy", "matplotlib")}}
        status("Saving separate SGP4, PINN, combined and comparison graphs and reports")
        from three_model_report import make_three_model_report
        meta["figures"] = make_three_model_report(folder, seconds, truth_r, truth_v, predictions, errors, metrics, meta,
                                                  {"PINN":pinn_history, "COMBINED":correction_history})
        for name in SOURCE_FILES:
            shutil.copy2(ROOT/name, folder/"source_code"/name)
        meta["runtime_seconds"] = time.perf_counter()-began
        write_json(folder/"summary.json", meta)
        status("Completed all three real model runs; replay verification follows")
        write_json(folder/"status.json", {"status":"complete", "run_id":folder.name})
        write_json(folder/"file_checksums.json", {p.relative_to(folder).as_posix():sha(p) for p in folder.rglob("*") if p.is_file()})
        from verify_three_model_run import verify
        verify(folder)
        write_json(output/"latest.json", {"run_id":folder.name})
        print(f"Results: {folder}", flush=True)
        # Original results stay immutable; publish a separately verified copy.
        try:
            from publish_model_results import publish_all
            publish_all(output)
            write_json(output/"dashboard_publication_status.json", {"run_id":folder.name, "status":"published"})
        except Exception as publication_error:
            write_json(output/"dashboard_publication_status.json", {"run_id":folder.name, "status":"failed", "error":str(publication_error)})
            print(f"Training and verification are complete, but dashboard publication failed: {publication_error}. Run publish_model_results.py to retry.", flush=True)
        return folder
    except Exception as exc:
        write_json(folder/"status.json", {"status":"failed", "error":str(exc), "run_id":folder.name})
        (folder/"error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pinn-steps", type=int, default=6000)
    parser.add_argument("--correction-steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--training-run", type=Path)
    args = parser.parse_args()
    run(args.pinn_steps, args.correction_steps, args.seed, args.input_dir, args.training_run)
