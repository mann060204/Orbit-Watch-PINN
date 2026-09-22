"""Run SGP4 and train PINN on one catalog; export a comparison without website updates."""
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

from comparison_evaluation import (flatten_metrics, match_encounters, physics_diagnostics,
                                   trajectory_details)
from evaluation import CLASSIFICATION_NAMES
from fetch_celestrak import (DEFAULT_GROUPS, fetch_group, parse_epoch, save_snapshot,
                            utc_now, utc_text, validate_records, write_json)
from orbit_engine import AnalysisConfig, prepare_catalog, propagate, reference_validation, screen
from pinn_device import inspect_device
from pinn_evaluation import pair_classification, trajectory_metrics
from pinn_model import LENGTH_KM, SPEED_KM_S, OrbitalPINN, normalize_states, rollout
from pinn_screening import screen_pinn
from run_analysis import write_csv
from train_pinn import split_objects, train

ROOT = Path(__file__).resolve().parent
SOURCE_FILES = ["run_comparison.py", "comparison_evaluation.py", "comparison_report.py", "fetch_celestrak.py",
                "orbit_engine.py", "run_analysis.py", "evaluation.py", "pinn_device.py", "pinn_model.py",
                "pinn_evaluation.py", "pinn_screening.py", "train_pinn.py", "requirements-lock.txt",
                "requirements-pinn.txt", "run_comparison.ps1"]
EVENT_FIELDS = ["event_id", "object1_id", "object2_id", "object1_name", "object2_name", "tca_utc",
                "miss_distance_km", "relative_speed_km_s", "collision_probability"]
MATCH_FIELDS = ["object1_id", "object2_id", "status", "pinn_event_id", "sgp4_event_id", "pinn_tca_utc",
                "sgp4_tca_utc", "pinn_miss_distance_km", "sgp4_miss_distance_km", "tca_difference_seconds",
                "miss_distance_difference_km", "relative_speed_difference_km_s"]


def website_hashes():
    paths = ["data/latest.json", "results/latest.json", "pinn_results/latest.json", "pinn_results/active.json",
             "dashboard_server.py", "dashboard/index.html", "dashboard/app.js", "dashboard/pinn.js"]
    return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() if (ROOT/name).exists() else None for name in paths}


def run_comparison(steps=6000, hours=1.0, segment_seconds=60.0, threshold_km=5.0,
                   seed=2026, snapshot=None, start=None, benchmark_repeats=5):
    if type(steps) is not int or not 100 <= steps <= 30000:
        raise ValueError("Use 100 to 30000 optimizer steps")
    if not np.isfinite(hours) or not .1 <= hours <= 6:
        raise ValueError("Use a forecast horizon from 0.1 to 6 hours")
    if not np.isfinite(segment_seconds) or not 10 <= segment_seconds <= 60:
        raise ValueError("Use neural steps from 10 to 60 seconds")
    config = AnalysisConfig(hours, segment_seconds, threshold_km, 72, False).validate()
    if type(benchmark_repeats) is not int or not 1 <= benchmark_repeats <= 10:
        raise ValueError("Use 1 to 10 benchmark repetitions")
    began = time.perf_counter()
    output = ROOT / "comparison_results"
    folder = output / utc_now().strftime("%Y%m%dT%H%M%S_%fZ")
    folder.mkdir(parents=True, exist_ok=False)
    for name in ("sgp4", "pinn", "evaluation", "figures", "source_code"):
        (folder/name).mkdir()
    website_before = website_hashes()
    def status(percent, message):
        print(f"{percent:5.1f}% {message}", flush=True)
        write_json(folder/"status.json", {"status":"running", "progress":percent, "message":message,
                                         "run_id":folder.name, "updated_at_utc":utc_text(utc_now())})
        with (folder/"run.log").open("a", encoding="utf-8") as stream:
            stream.write(f"{utc_text(utc_now())} | {percent:.1f}% | {message}\n")
    try:
        status(1, "Preparing one CelesTrak snapshot for both models (two-hour source cache applies)")
        if snapshot:
            source = Path(snapshot).resolve()
            records = validate_records(json.loads((source/"orbital_data.json").read_text(encoding="utf-8")))
            manifest = json.loads((source/"manifest.json").read_text(encoding="utf-8"))
            shutil.copytree(source, folder/"input_snapshot")
            source = folder/"input_snapshot"
        else:
            datasets = [fetch_group(group, ROOT/"data") for group in DEFAULT_GROUPS]
            source, manifest = save_snapshot(datasets, folder/"data")
            records = json.loads((source/"orbital_data.json").read_text(encoding="utf-8"))
        start = start or utc_now()
        hardware = inspect_device()
        torch.set_num_threads(hardware["torch_threads"])
        device = hardware["selected_device"]
        write_json(folder/"hardware.json", hardware)
        write_json(folder/"config.json", {**config.to_dict(), "steps":steps, "seed":seed,
                   "start_utc":utc_text(start), "device":device, "benchmark_repeats":benchmark_repeats,
                   "input_snapshot":source.relative_to(folder).as_posix(), "website_publication":False})
        satellites, rows, excluded = prepare_catalog(records)
        errors0, r0, v0 = propagate(satellites, start, [0.0])
        selected = []
        for i, row in enumerate(rows):
            age = (start-parse_epoch(row["EPOCH"])).total_seconds()/3600
            reason = ("Epoch outside 72-hour age limit" if abs(age)>72 else
                      "Invalid SGP4 initial state" if errors0[i, 0] or not np.isfinite(np.r_[r0[i, 0], v0[i, 0]]).all() else None)
            if reason:
                excluded.append({"norad_id":int(row["NORAD_CAT_ID"]), "reason":reason, "epoch_age_hours":age})
            else:
                selected.append(i)
        rows = [rows[i] for i in selected]
        satellites = [satellites[i] for i in selected]
        if len(rows)<30:
            raise ValueError("Fewer than 30 fresh valid objects; fetch a current catalog")
        ids = np.array([int(row["NORAD_CAT_ID"]) for row in rows])
        initial = normalize_states(r0[selected, 0], v0[selected, 0])
        partitions = split_objects(rows, seed)
        if any(len(indices)==0 for indices in partitions.values()):
            raise ValueError("Every grouped object split must be nonempty")
        write_json(folder/"excluded_objects.json", excluded)
        write_json(folder/"eligible_objects.json", rows)
        write_json(folder/"data_splits.json", {name:ids[indices].tolist() for name, indices in partitions.items()})
        np.savez_compressed(folder/"initial_conditions.npz", norad_ids=ids, normalized_state=initial,
                            start_utc=np.array(utc_text(start)))
        def training_progress(fraction, message, row):
            status(5+60*fraction, message)
            with (folder/"pinn/training_history.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row)+"\n")
        status(4, f"Actual PINN training: {len(rows):,} shared objects; grouped training/validation/test split")
        model, untrained, history, proof = train(initial, partitions, hardware, steps, segment_seconds, seed,
                                                training_progress, folder/"pinn/model.pt")
        write_json(folder/"pinn/training_proof.json", proof)
        write_csv(folder/"pinn/training_history.csv", history)
        seconds = np.append(np.arange(0, hours*3600, segment_seconds, dtype=float), hours*3600)
        def synchronize():
            if device == "cuda":
                torch.cuda.synchronize()
        def evaluate_model(name):
            if name == "SGP4":
                return propagate(satellites, start, seconds)
            return rollout(model, initial, seconds, device)
        status(68, "Training finished; computing future SGP4 references and neural predictions on the same grid")
        # Both are warmed once, then timed repeatedly in alternating order. No I/O in timed regions.
        for name in ("SGP4", "PINN"):
            evaluate_model(name)
            synchronize()
        timing_rows, results = [], {}
        for repeat in range(benchmark_repeats):
            for order, name in enumerate(("SGP4", "PINN") if repeat%2==0 else ("PINN", "SGP4")):
                synchronize()
                before = time.perf_counter()
                result = evaluate_model(name)
                synchronize()
                elapsed = time.perf_counter()-before
                timing_rows.append({"model":name, "repeat":repeat+1, "order_in_repeat":order+1,
                                    "propagation_seconds":elapsed, "future_states":len(ids)*(len(seconds)-1)})
                results[name] = result
        errors, sgp4_r, sgp4_v = results["SGP4"]
        prediction = results["PINN"]
        pinn_r, pinn_v = prediction[:, :, :3]*LENGTH_KM, prediction[:, :, 3:]*SPEED_KM_S
        if (errors != 0).any() or not np.isfinite(sgp4_r).all() or not np.isfinite(sgp4_v).all() or not np.isfinite(prediction).all():
            raise ValueError("A model produced an invalid future state; a complete comparison cannot be reported")
        for name, r, v in (("sgp4", sgp4_r, sgp4_v), ("pinn", pinn_r, pinn_v)):
            np.savez_compressed(folder/name/"trajectories.npz", norad_ids=ids, seconds_from_start=seconds,
                                start_utc=np.array(utc_text(start)), position_teme_km=r, velocity_teme_km_s=v)
        np.savez_compressed(folder/"sgp4/error_codes.npz", norad_ids=ids, error_codes=errors)
        write_csv(folder/"evaluation/propagation_benchmarks.csv", timing_rows)
        status(73, "Screening SGP4 and PINN independently with identical objects and threshold")
        before = time.perf_counter()
        sgp4_events, sgp4_audit, screen_excluded, refinement_errors, sgp4_stats = screen(satellites, rows, sgp4_r, errors, start, seconds, config)
        sgp4_screen_seconds = time.perf_counter()-before
        if screen_excluded or refinement_errors:
            raise ValueError("SGP4 screening changed the common object set or failed refinement")
        shared = sgp4_stats["shared_orbit_pairs"]
        before = time.perf_counter()
        pinn_events, pinn_audit, pinn_stats = screen_pinn(model, prediction, rows, start, seconds, threshold_km, shared, device,
                                                        lambda message:status(79, message))
        synchronize()
        pinn_screen_seconds = time.perf_counter()-before
        for name, events in (("sgp4", sgp4_events), ("pinn", pinn_events)):
            write_json(folder/name/"events.json", events)
            write_csv(folder/name/"events.csv", events, EVENT_FIELDS)
        write_json(folder/"sgp4/screening_diagnostics.json", sgp4_stats)
        write_json(folder/"sgp4/candidate_audit.json", sgp4_audit)
        write_json(folder/"pinn/screening_diagnostics.json", pinn_stats)
        write_json(folder/"pinn/candidate_audit.json", pinn_audit)
        write_json(folder/"shared_orbit_pairs.json", shared)
        status(85, "Calculating trajectory, encounter, timing, and physics evaluations")
        trajectory, _, _ = trajectory_metrics(prediction, sgp4_r, sgp4_v, partitions)
        extra, time_rows, object_rows, pe, ve, rtn = trajectory_details(pinn_r, pinn_v, sgp4_r, sgp4_v, ids, rows, partitions, seconds)
        classification, pair_rows = pair_classification(pinn_events, sgp4_events, ids, shared)
        heldout, _ = pair_classification(pinn_events, sgp4_events, ids[partitions["test"]], shared)
        encounter_metrics, event_rows = match_encounters(pinn_events, sgp4_events, 2*segment_seconds)
        write_csv(folder/"evaluation/per_object_errors.csv", object_rows)
        write_csv(folder/"evaluation/errors_by_forecast_time.csv", time_rows)
        write_csv(folder/"evaluation/pair_agreement.csv", pair_rows, ["object1_id", "object2_id", "pinn_positive", "sgp4_reference_positive", "outcome"])
        write_csv(folder/"evaluation/event_matches.csv", event_rows, MATCH_FIELDS)
        write_json(folder/"evaluation/pair_universe.json", {"norad_ids":ids.tolist(), "excluded_shared_pairs":shared,
                   "definition":"All unordered pairs of these IDs minus shared modeled-orbit pairs. Pairs absent from pair_agreement.csv are negative in both model screenings; they are not observed negatives."})
        np.savez_compressed(folder/"evaluation/trajectory_differences.npz", norad_ids=ids, seconds_from_start=seconds,
                            position_vector_difference_km=pe, velocity_vector_difference_km_s=ve, signed_position_rtn_difference_km=rtn)
        timing = {}
        for name, events, screening_time in (("SGP4", sgp4_events, sgp4_screen_seconds), ("PINN", pinn_events, pinn_screen_seconds)):
            values = [row["propagation_seconds"] for row in timing_rows if row["model"]==name]
            timing[name] = {"training_seconds":proof["training_seconds"] if name=="PINN" else 0.0,
                            "training_applicable":name=="PINN", "propagation_seconds_median":float(np.median(values)),
                            "propagation_seconds_min":min(values), "propagation_seconds_max":max(values),
                            "screening_seconds":screening_time, "close_approach_events":len(events),
                            "closest_approach_km":events[0]["miss_distance_km"] if events else None,
                            "valid_future_states":int(len(ids)*(len(seconds)-1)), "finite_state_fraction":1.0}
        timing["pinn_over_sgp4_propagation_time_ratio"] = timing["PINN"]["propagation_seconds_median"]/timing["SGP4"]["propagation_seconds_median"]
        timing["method"] = "One warm-up each, alternating order, perf_counter, CUDA synchronization when needed; array computation and device transfers included, disk I/O excluded. Screen timing includes each existing routine's diagnostics (SGP4 also computes distance curves)."
        unavailable = {"status":"not_evaluable", **{name:None for name in CLASSIFICATION_NAMES},
                       "position_rmse_km":None, "velocity_rmse_km_s":None, "roc_auc":None, "pr_auc":None,
                       "brier_score":None, "collision_probability":None,
                       "reason":"No independent observed trajectories or labeled collision outcomes. No encounter covariances, radii, or calibrated probability scores."}
        metrics = {"model_comparison":timing, "pinn_vs_sgp4_trajectory":trajectory,
                   "trajectory_components":extra, "pinn_vs_sgp4_all_pairs":classification,
                   "pinn_vs_sgp4_test_only_pairs":heldout, "one_to_one_event_agreement":encounter_metrics,
                   "physics_diagnostics":{"SGP4":physics_diagnostics(sgp4_r, sgp4_v, seconds),
                                          "PINN":physics_diagnostics(pinn_r, pinn_v, seconds)},
                   "observational_evaluation":{"SGP4":dict(unavailable), "PINN":dict(unavailable)},
                   "sgp4_implementation_reference_check":reference_validation(), "pinn_training":proof}
        write_json(folder/"evaluation/metrics.json", metrics)
        write_csv(folder/"evaluation/all_metrics.csv", flatten_metrics(metrics), ["metric", "value", "status"])
        write_csv(folder/"evaluation/model_comparison.csv", [{"model":name, **timing[name]} for name in ("SGP4", "PINN")])
        status(92, "Reloading saved neural weights and verifying comparison artifacts")
        checkpoint = torch.load(folder/"pinn/model.pt", map_location="cpu", weights_only=True)
        restored = OrbitalPINN(checkpoint["width"], checkpoint["depth"]).double().eval()
        restored.load_state_dict(checkpoint["model_state_dict"])
        replay = rollout(restored, initial, seconds)
        replay_max_km = float(np.max(np.abs(replay[:, :, :3]*LENGTH_KM-pinn_r)))
        sets = [set(ids[indices].tolist()) for indices in partitions.values()]
        cm = classification["confusion_matrix"]
        test_rmse = float(np.sqrt(np.mean(np.linalg.norm(pinn_r[partitions["test"], 1:]-sgp4_r[partitions["test"], 1:], axis=-1)**2)))
        verification = {"checkpoint_replay_maximum_position_difference_km":replay_max_km,
                        "checkpoint_replay_passed":replay_max_km<1e-7,
                        "object_splits_disjoint":all(not a&b for i, a in enumerate(sets) for b in sets[i+1:]),
                        "same_initial_positions":bool(np.allclose(pinn_r[:, 0], sgp4_r[:, 0], rtol=0, atol=1e-10)),
                        "pair_confusion_sums_to_universe":sum(cm.values())==classification["pair_universe"],
                        "test_rmse_recomputed":abs(test_rmse-trajectory["test"]["position_vector_rmse_km"])<1e-10,
                        "no_invented_probability":all(event["collision_probability"] is None for event in pinn_events+sgp4_events),
                        "website_files_and_result_pointers_unchanged":website_before==website_hashes()}
        if not all(value for key, value in verification.items() if key != "checkpoint_replay_maximum_position_difference_km"):
            raise ValueError(f"Comparison artifact checks failed: {verification}")
        write_json(folder/"evaluation/artifact_verification.json", verification)
        hashes = {}
        for name in SOURCE_FILES:
            shutil.copy2(ROOT/name, folder/"source_code"/name)
            hashes[name] = hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        summary = {"run_id":folder.name, "window_start_utc":utc_text(start), "window_end_utc":utc_text(start+timedelta(hours=hours)),
                   "config":config.to_dict(), "input_manifest":manifest, "input_snapshot":source.relative_to(folder).as_posix(),
                   "input_objects":len(records), "modeled_objects":len(ids), "excluded_objects":len(excluded),
                   "shared_orbit_pairs":len(shared), "split_counts":{name:len(indices) for name, indices in partitions.items()},
                   "hardware":hardware, "metrics":metrics, "source_sha256":hashes,
                   "package_versions":{name:version(name) for name in ("numpy", "scipy", "sgp4", "torch", "matplotlib")},
                   "website_publication":False}
        from comparison_report import make_comparison_report
        make_comparison_report(folder, summary, history, time_rows, pe, event_rows)
        summary["total_runtime_seconds"] = time.perf_counter()-began
        write_json(folder/"summary.json", summary)
        status(100, f"Saved folder-only comparison: SGP4 {len(sgp4_events)} events, PINN {len(pinn_events)} events")
        write_json(folder/"status.json", {"status":"complete", "run_id":folder.name, "updated_at_utc":utc_text(utc_now())})
        file_hashes = {path.relative_to(folder).as_posix():hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in folder.rglob("*") if path.is_file()}
        write_json(folder/"file_checksums.json", file_hashes)
        write_json(output/"latest.json", {"run_id":folder.name})
        print(f"Comparison results: {folder}", flush=True)
        return folder, summary
    except Exception as exc:
        write_json(folder/"status.json", {"status":"failed", "run_id":folder.name, "error":str(exc)})
        (folder/"error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--hours", type=float, default=1)
    parser.add_argument("--segment-seconds", type=float, default=60)
    parser.add_argument("--threshold-km", type=float, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--snapshot", type=Path, help="Exact existing snapshot folder; bypasses downloads")
    parser.add_argument("--start", help="Explicit UTC start for reproduction")
    parser.add_argument("--benchmark-repeats", type=int, default=5)
    args = parser.parse_args()
    run_comparison(args.steps, args.hours, args.segment_seconds, args.threshold_km, args.seed,
                   args.snapshot, parse_epoch(args.start) if args.start else None, args.benchmark_repeats)
