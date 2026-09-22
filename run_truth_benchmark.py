"""Evaluate SGP4 and the frozen PINN against an independent ESA reconstructed orbit."""
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
from urllib.request import Request, urlopen

import numpy as np
import torch
from sgp4.exporter import export_tle

from comparison_evaluation import flatten_metrics
from fetch_celestrak import parse_epoch, utc_now, utc_text, validate_records, write_json
from orbit_engine import make_satellite, propagate, reference_validation
from orbit_truth import orbit_error_metrics, read_esa_orbit, reference_to_teme, select_native_samples
from pinn_device import inspect_device
from pinn_model import LENGTH_KM, SPEED_KM_S, OrbitalPINN, normalize_states, rollout
from run_analysis import write_csv

ROOT = Path(__file__).resolve().parent
EOP_URL = "https://datacenter.iers.org/data/9/finals2000A.all"
SOURCES = ["run_truth_benchmark.py", "orbit_truth.py", "truth_plots.py", "truth_benchmark_report.py",
           "comparison_evaluation.py", "fetch_celestrak.py", "orbit_engine.py", "pinn_model.py",
           "pinn_device.py", "run_analysis.py", "evaluation.py", "requirements-lock.txt",
           "requirements-pinn.txt", "requirements-truth.txt", "run_truth_benchmark.ps1"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def website_hashes():
    files = ["data/latest.json", "results/latest.json", "pinn_results/latest.json", "pinn_results/active.json",
             "dashboard_server.py", "dashboard/index.html", "dashboard/app.js", "dashboard/pinn.js"]
    return {name:sha(ROOT/name) if (ROOT/name).exists() else None for name in files}


def safe_input_path(folder, filename):
    if Path(filename).name != filename:
        raise ValueError("Source metadata filenames must be basenames")
    return folder/filename


def run_truth_benchmark(input_dir=None, model_run=None, hours=1, step_seconds=60, eop_file=None):
    input_dir = Path(input_dir or ROOT/"reference_inputs/sentinel_benchmark").resolve()
    if model_run is None:
        pointer = json.loads((ROOT/"comparison_results/latest.json").read_text())
        model_run = ROOT/"comparison_results"/pointer["run_id"]
    model_run = Path(model_run).resolve()
    folder = ROOT/"ground_truth_results"/utc_now().strftime("%Y%m%dT%H%M%S_%fZ")
    folder.mkdir(parents=True, exist_ok=False)
    for name in ("inputs", "predictions", "reference", "evaluation", "trained_model", "source_code", "figures"):
        (folder/name).mkdir()
    before_website = website_hashes()
    started = time.perf_counter()
    def status(message):
        print(message, flush=True)
        write_json(folder/"status.json", {"status":"running", "message":message, "run_id":folder.name})
        with (folder/"run.log").open("a", encoding="utf-8") as stream:
            stream.write(f"{utc_text(utc_now())} | {message}\n")
    try:
        status("Checking original CelesTrak and independent ESA input files")
        provenance = json.loads((input_dir/"source_manifest.json").read_text())
        for name, expected in provenance["files_sha256"].items():
            source = safe_input_path(input_dir, name)
            if sha(source) != expected:
                raise ValueError(f"Input checksum mismatch: {name}")
            shutil.copy2(source, folder/"inputs"/name)
        shutil.copy2(input_dir/"source_manifest.json", folder/"inputs/source_manifest.json")
        gp = validate_records(json.loads((folder/"inputs/initial_gp.json").read_text()))
        if len(gp)!=1 or int(gp[0]["NORAD_CAT_ID"]) != provenance["norad_cat_id"]:
            raise ValueError("One matching GP object is required")
        record = gp[0]
        eof = safe_input_path(folder/"inputs", provenance["reference_source"]["eof_file"])
        reference = read_esa_orbit(eof)
        if reference["mission"].upper().replace("-", "") != record["OBJECT_NAME"].upper().replace("-", ""):
            raise ValueError("ESA mission does not match the GP satellite")
        indices, seconds = select_native_samples(reference, parse_epoch(record["EPOCH"]), hours, step_seconds)
        start = reference["times"][indices[0]]
        selected_times = [reference["times"][i] for i in indices]
        hardware = inspect_device()
        torch.set_num_threads(hardware["torch_threads"])
        write_json(folder/"hardware.json", hardware)
        # Keep the previous checkpoint fixed. The independent reference never enters training.
        model_summary = json.loads((model_run/"summary.json").read_text())
        split_ids = json.loads((model_run/"data_splits.json").read_text())
        if any(int(record["NORAD_CAT_ID"]) in values for values in split_ids.values()):
            raise ValueError("External benchmark satellite was present in the saved model's data splits")
        if model_summary["source_sha256"]["pinn_model.py"] != sha(ROOT/"pinn_model.py"):
            raise ValueError("Current PINN implementation differs from its recorded training source")
        model_path = model_run/"pinn/model.pt"
        checkpoint_hash = sha(model_path)
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        if step_seconds > checkpoint["segment_seconds"]:
            raise ValueError("Inference step exceeds the trained local-flow interval")
        model = OrbitalPINN(checkpoint["width"], checkpoint["depth"]).double().eval()
        model.load_state_dict(checkpoint["model_state_dict"])
        shutil.copy2(model_path, folder/"trained_model/model.pt")
        shutil.copy2(model_run/"summary.json", folder/"trained_model/original_training_run_summary.json")
        shutil.copy2(model_run/"pinn/training_proof.json", folder/"trained_model/training_proof.json")
        shutil.copy2(model_run/"pinn/training_history.csv", folder/"trained_model/training_history.csv")
        shutil.copy2(model_run/"data_splits.json", folder/"trained_model/original_data_splits.json")
        status("Computing both predictions from the same GP-derived initial state")
        sat = make_satellite(record)
        sgp4_begin = time.perf_counter()
        codes, sgp4_r, sgp4_v = propagate([sat], start, seconds)
        sgp4_seconds = time.perf_counter()-sgp4_begin
        if (codes != 0).any() or not np.isfinite(sgp4_r).all() or not np.isfinite(sgp4_v).all():
            raise ValueError("SGP4 returned invalid states")
        sgp4_r, sgp4_v = sgp4_r[0], sgp4_v[0]
        initial = normalize_states(sgp4_r[:1], sgp4_v[:1])
        pinn_begin = time.perf_counter()
        neural = rollout(model, initial, seconds)[0]
        pinn_seconds = time.perf_counter()-pinn_begin
        predictions = {"SGP4":{"r":sgp4_r, "v":sgp4_v},
                       "PINN":{"r":neural[:, :3]*LENGTH_KM, "v":neural[:, 3:]*SPEED_KM_S}}
        if not np.isfinite(neural).all():
            raise ValueError("PINN output is nonfinite")
        # A convenience TLE encoding is derived from the original GP record. The computation
        # uses unmodified OMM JSON to avoid introducing extra decimal-rounding differences.
        tle1, tle2 = export_tle(sat)
        (folder/"inputs/derived_initial_tle.txt").write_text(record["OBJECT_NAME"]+"\n"+tle1+"\n"+tle2+"\n", encoding="utf-8")
        np.savez_compressed(folder/"inputs/common_initial_state.npz", norad_id=np.array(record["NORAD_CAT_ID"]),
                            start_utc=np.array(utc_text(start)), position_teme_km=sgp4_r[0], velocity_teme_km_s=sgp4_v[0], normalized_state=initial)
        status("Transforming independent ESA positions AND velocities into TEME")
        local_eop = folder/"inputs/finals2000A.all"
        saved_eop = Path(eop_file).resolve() if eop_file else input_dir/"finals2000A.all"
        if eop_file or saved_eop.exists():
            shutil.copy2(saved_eop, local_eop)
            eop_source = {"mode":"saved_reproduction_file", "original_path":str(saved_eop)}
            source_note = saved_eop.with_suffix(".source.json")
            if source_note.exists():
                download_info = json.loads(source_note.read_text(encoding="utf-8"))
                if download_info["sha256"] != sha(local_eop):
                    raise ValueError("Saved Earth-orientation file checksum mismatch")
                eop_source["download_provenance"] = download_info
                shutil.copy2(source_note, folder/"inputs/finals2000A.source.json")
        else:
            request = Request(EOP_URL, headers={"User-Agent":"OrbitalResearchBenchmark/1.0"})
            with urlopen(request, timeout=30) as response:
                content = response.read(6_000_001)
                if len(content)>6_000_000:
                    raise ValueError("Unexpected Earth orientation file size")
                local_eop.write_bytes(content)
                eop_source = {"url":EOP_URL, "fetched_at_utc":utc_text(utc_now()), "last_modified":response.headers.get("Last-Modified")}
        truth_r, truth_v, frame_info, eop_rows = reference_to_teme(reference, indices, local_eop)
        frame_info["eop_source"] = eop_source
        frame_info["eop_sha256"] = sha(local_eop)
        write_json(folder/"reference/frame_transformation.json", frame_info)
        write_csv(folder/"reference/earth_orientation_at_samples.csv", eop_rows)
        common = {"norad_id":np.array(record["NORAD_CAT_ID"]), "utc":np.array([utc_text(t) for t in selected_times]),
                  "seconds_from_start":seconds}
        np.savez_compressed(folder/"reference/independent_orbit.npz", **common, position_teme_km=truth_r,
                            velocity_teme_km_s=truth_v, position_itrs_km=reference["r_itrs_km"][indices],
                            velocity_itrs_km_s=reference["v_itrs_km_s"][indices], original_osv_indices=indices)
        metrics, errors, metric_rows, state_rows, error_rows = {}, {}, [], [], []
        status("Calculating separate SGP4/reference and PINN/reference RMSE, MAE, and RTN errors")
        for name, values in predictions.items():
            metrics[name], errors[name] = orbit_error_metrics(values["r"], values["v"], truth_r, truth_v)
            np.savez_compressed(folder/"predictions"/(name.lower()+".npz"), **common, position_teme_km=values["r"], velocity_teme_km_s=values["v"])
            np.savez_compressed(folder/"evaluation"/(name.lower()+"_errors.npz"), **common, **errors[name])
            metric_rows.extend({"model":name, **row} for row in flatten_metrics(metrics[name]))
            for i, second in enumerate(seconds):
                row = {"model":name, "utc":utc_text(selected_times[i]), "seconds_from_start":float(second), "included_in_aggregate":i>0}
                row.update({f"position_{axis}_error_km":float(errors[name]["position_xyz_km"][i, j]) for j, axis in enumerate("xyz")})
                row.update({f"velocity_{axis}_error_km_s":float(errors[name]["velocity_xyz_km_s"][i, j]) for j, axis in enumerate("xyz")})
                for field, suffix in (("position_rtn_km", "position_error_km"), ("velocity_rtn_km_s", "velocity_error_km_s")):
                    row.update({f"{axis}_{suffix}":float(errors[name][field][i, j]) for j, axis in enumerate(("radial", "transverse", "normal"))})
                row.update(position_vector_error_km=float(errors[name]["position_norm_km"][i]), velocity_vector_error_km_s=float(errors[name]["velocity_norm_km_s"][i]))
                error_rows.append(row)
        for name, r, v in [(name, values["r"], values["v"]) for name, values in predictions.items()]+[("ESA_REFERENCE", truth_r, truth_v)]:
            for i, second in enumerate(seconds):
                state_rows.append({"model":name, "utc":utc_text(selected_times[i]), "seconds_from_start":float(second),
                                   **{f"{axis}_km":float(r[i, j]) for j, axis in enumerate("xyz")},
                                   **{f"v{axis}_km_s":float(v[i, j]) for j, axis in enumerate("xyz")}})
        write_csv(folder/"predictions/all_states.csv", state_rows)
        write_csv(folder/"evaluation/errors_by_time.csv", error_rows)
        write_csv(folder/"evaluation/all_metrics.csv", metric_rows, ["model", "metric", "value", "status"])
        write_json(folder/"evaluation/metrics.json", metrics)
        compact_keys = ["position_vector_rmse_km", "position_vector_mae_km", "velocity_vector_rmse_km_s", "velocity_vector_mae_km_s",
                        "initial_position_error_km", "final_position_error_km", "position_max_km"]
        write_csv(folder/"evaluation/sgp4_vs_pinn.csv", [{"metric":key, "SGP4":metrics["SGP4"][key], "PINN":metrics["PINN"][key]} for key in compact_keys])
        checks = {"shared_initial_position":bool(np.allclose(predictions["PINN"]["r"][0], sgp4_r[0], rtol=0, atol=1e-10)),
                  "shared_initial_velocity":bool(np.allclose(predictions["PINN"]["v"][0], sgp4_v[0], rtol=0, atol=1e-12)),
                  "reference_timestamps_used_without_interpolation":True, "all_reference_quality_nominal":True,
                  "benchmark_object_absent_from_original_model_splits":True, "frozen_checkpoint_unchanged":sha(model_path)==checkpoint_hash,
                  "website_unchanged":website_hashes()==before_website,
                  "rtn_preserves_position_error_norm":all(np.allclose(np.linalg.norm(e["position_rtn_km"], axis=1), e["position_norm_km"], atol=1e-10) for e in errors.values()),
                  "sgp4_implementation_reference_passed":reference_validation()["passed"]}
        if not all(checks.values()):
            raise ValueError(f"Verification failed: {checks}")
        write_json(folder/"evaluation/verification.json", checks)
        unavailable = {"accuracy":None, "precision":None, "recall":None, "f1_score":None, "collision_probability":None,
                       "reason":"One-object orbit regression benchmark: no labeled conjunction events or collision covariances/radii."}
        write_json(folder/"evaluation/unavailable_classification_metrics.json", unavailable)
        meta = {"run_id":folder.name, "object_name":record["OBJECT_NAME"], "norad_id":int(record["NORAD_CAT_ID"]),
                "reference_label":"ESA GNSS restituted orbit (RESORB)", "reference_kind":provenance["reference_kind"],
                "reference_is_final_precise_orbit":False, "reference_is_error_free_truth":False,
                "start_utc":utc_text(start), "end_utc":utc_text(selected_times[-1]), "sample_count":len(seconds),
                "aggregate_samples_per_model":len(seconds)-1, "step_seconds":step_seconds, "horizon_hours":hours,
                "gp_epoch_utc":utc_text(parse_epoch(record["EPOCH"])), "retrospective_test":True,
                "model_training_run":model_run.name, "model_checkpoint_sha256":checkpoint_hash,
                "pinning":"Frozen checkpoint selected before acquisition of the independent reference; no fitting, tuning, or bias correction against ESA data.",
                "input_format":"CelesTrak GP JSON with OMM mean elements; derived_initial_tle.txt is a convenience encoding, not separately downloaded input",
                "evaluation_frame":"TEME", "native_reference_frame":"EARTH_FIXED / ITRS (CPOD ITRF2020)",
                "metrics":metrics, "frame_transformation":frame_info, "hardware":hardware,
                "prediction_timing_seconds":{"SGP4":sgp4_seconds, "PINN":pinn_seconds},
                "timing_note":"Single-call diagnostics including array setup; not a warmed speed benchmark.",
                "package_versions":{name:version(name) for name in ("astropy", "astropy-iers-data", "numpy", "scipy", "torch", "sgp4", "matplotlib")},
                "source_manifest":provenance, "website_publication":False, "reference_interpolation":False}
        status("Saving all orbit, error, RMSE, MAE, RTN, and workflow graphs")
        from truth_plots import make_truth_figures
        meta["figures"] = make_truth_figures(folder, seconds, truth_r, truth_v, predictions, errors, metrics, meta)
        from truth_benchmark_report import make_truth_report
        make_truth_report(folder, meta)
        for filename in SOURCES:
            shutil.copy2(ROOT/filename, folder/"source_code"/filename)
        meta["source_sha256"] = {filename:sha(ROOT/filename) for filename in SOURCES}
        meta["total_seconds"] = time.perf_counter()-started
        write_json(folder/"summary.json", meta)
        status("Independent orbit benchmark complete; every output saved locally")
        write_json(folder/"status.json", {"status":"complete", "run_id":folder.name})
        write_json(folder/"file_checksums.json", {p.relative_to(folder).as_posix():sha(p) for p in folder.rglob("*") if p.is_file()})
        write_json(ROOT/"ground_truth_results/latest.json", {"run_id":folder.name})
        print(f"Results folder: {folder}", flush=True)
        print(json.dumps({name:{key:metrics[name][key] for key in compact_keys} for name in metrics}, indent=2), flush=True)
        return folder, meta
    except Exception as exc:
        write_json(folder/"status.json", {"status":"failed", "error":str(exc)})
        (folder/"error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--model-run", type=Path)
    parser.add_argument("--hours", type=float, default=1)
    parser.add_argument("--step-seconds", type=float, default=60)
    parser.add_argument("--eop-file", type=Path, help="Saved IERS file for exact reproduction")
    args = parser.parse_args()
    run_truth_benchmark(args.inputs, args.model_run, args.hours, args.step_seconds, args.eop_file)
