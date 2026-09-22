"""Independently audit a completed orbit benchmark without training or network access.

Only the verifier's JSON/source copy and their checksum entries are written after
all original artifact hashes and numerical checks pass. No benchmark values are
rewritten. Run with the project virtual environment and a completed result folder.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET
import zipfile

import numpy as np

sys.dont_write_bytecode = True
OWNED = ("evaluation/artifact_verification.json", "source_code/verify_truth_run.py")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def utc(value):
    return datetime.fromisoformat(str(value).removeprefix("UTC=").replace("Z", "+00:00")).replace(tzinfo=timezone.utc)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, label, atol=1e-10, rtol=1e-11):
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, err_msg=label)


def native_reference(path):
    """Parse XML independently of orbit_truth.read_esa_orbit."""
    document = ET.parse(path).getroot()
    require(document.findtext(".//Variable_Header/Ref_Frame") == "EARTH_FIXED", "ESA frame")
    require(document.findtext(".//Variable_Header/Time_Reference") == "UTC", "ESA time system")
    osvs = document.findall(".//List_of_OSVs/OSV")
    require(len(osvs) == int(document.find(".//List_of_OSVs").get("count")), "ESA OSV count")
    dates = [utc(osv.findtext("UTC")) for osv in osvs]
    ut1 = [utc(osv.findtext("UT1").removeprefix("UT1=")) for osv in osvs]
    for osv in osvs:
        for coordinate in ("X", "Y", "Z", "VX", "VY", "VZ"):
            require(osv.find(coordinate).get("unit") == ("m/s" if coordinate.startswith("V") else "m"), "ESA units")
    r = np.asarray([[float(osv.findtext(key))/1000 for key in ("X", "Y", "Z")] for osv in osvs])
    v = np.asarray([[float(osv.findtext(key))/1000 for key in ("VX", "VY", "VZ")] for osv in osvs])
    return dates, r, v, np.asarray([(a-b).total_seconds() for a, b in zip(ut1, dates)]), osvs


def transform_reference(dates, r, v, dut1, eop_file):
    """Use Astropy directly, without the benchmark's frame-conversion wrapper."""
    from astropy import units as u
    from astropy.coordinates import CartesianDifferential, CartesianRepresentation, ITRS, TEME
    from astropy.time import Time
    from astropy.utils import iers
    iers.conf.auto_download = False
    iers.conf.iers_degraded_accuracy = "error"
    table = iers.IERS_A.open(str(eop_file))
    t = Time(dates, scale="utc")
    t.delta_ut1_utc = dut1
    _, _, flag = table.pm_xy(t, return_status=True)
    require(np.all(flag >= 0), "EOP coverage")
    cartesian = CartesianRepresentation(r.T*u.km, differentials=CartesianDifferential(v.T*u.km/u.s))
    with iers.earth_orientation_table.set(table):
        state = ITRS(cartesian, obstime=t).transform_to(TEME(obstime=t)).cartesian
    return state.xyz.to_value(u.km).T, state.differentials["s"].d_xyz.to_value(u.km/u.s).T, flag


def independent_metrics(predicted_r, predicted_v, reference_r, reference_v):
    """Recompute formulas without importing the benchmark metric function."""
    r_axis = reference_r/np.linalg.norm(reference_r, axis=1)[:, None]
    n_axis = np.cross(reference_r, reference_v)
    n_axis /= np.linalg.norm(n_axis, axis=1)[:, None]
    t_axis = np.cross(n_axis, r_axis)
    basis = np.stack([r_axis, t_axis, n_axis], axis=2)
    delta_r, delta_v = predicted_r-reference_r, predicted_v-reference_v
    rtn_r = np.matmul(delta_r[:, None, :], basis)[:, 0, :]
    rtn_v = np.matmul(delta_v[:, None, :], basis)[:, 0, :]
    pnorm, vnorm = np.linalg.norm(delta_r, axis=1), np.linalg.norm(delta_v, axis=1)
    result = {"future_samples":len(pnorm)-1}
    for label, norms, unit in (("position", pnorm, "km"), ("velocity", vnorm, "km_s")):
        values = norms[1:]
        result.update({f"initial_{label}_error_{unit}":norms[0], f"final_{label}_error_{unit}":norms[-1],
                       f"{label}_vector_rmse_{unit}":np.sqrt(np.dot(values, values)/len(values)),
                       f"{label}_vector_mae_{unit}":sum(values)/len(values),
                       f"{label}_max_{unit}":max(values), f"{label}_p95_{unit}":np.percentile(values, 95)})
    for label, values, unit in (("position_xyz", delta_r, "km"), ("velocity_xyz", delta_v, "km_s"),
                                 ("position_rtn", rtn_r, "km"), ("velocity_rtn", rtn_v, "km_s")):
        result[f"{label}_rmse_{unit}"] = [np.sqrt(np.dot(column, column)/len(column)) for column in values[1:].T]
        result[f"{label}_mae_{unit}"] = [sum(abs(column))/len(column) for column in values[1:].T]
    result["position_rtn_bias_km"] = np.sum(rtn_r[1:], axis=0)/(len(rtn_r)-1)
    result["position_xyz_bias_km"] = np.sum(delta_r[1:], axis=0)/(len(delta_r)-1)
    close(np.linalg.norm(rtn_r, axis=1), pnorm, "RTN position norm")
    close(np.linalg.norm(rtn_v, axis=1), vnorm, "RTN velocity norm")
    return result, {"position_xyz_km":delta_r, "velocity_xyz_km_s":delta_v,
                    "position_rtn_km":rtn_r, "velocity_rtn_km_s":rtn_v,
                    "position_norm_km":pnorm, "velocity_norm_km_s":vnorm}


def verify(folder):
    folder = Path(folder).resolve()
    manifest_path = folder/"file_checksums.json"
    manifest_bytes = manifest_path.read_bytes()
    hashes = read_json(manifest_path)
    for relative, expected in hashes.items():
        path = (folder/relative).resolve()
        require(path.is_relative_to(folder), f"Checksum path escaped folder: {relative}")
        require(path.is_file() and digest(path) == expected, f"Checksum mismatch: {relative}")
    files = {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}
    require(files == set(hashes) | {"file_checksums.json"}, "Untracked or missing original artifact files")
    summary = read_json(folder/"summary.json")
    require(read_json(folder/"status.json")["status"] == "complete", "Run is incomplete")
    provenance = read_json(folder/"inputs/source_manifest.json")
    require(provenance == summary["source_manifest"], "Summary source manifest differs")
    for name, expected in provenance["files_sha256"].items():
        require(digest(folder/"inputs"/name) == expected, f"Original input mismatch: {name}")
    for name, expected in summary["source_sha256"].items():
        require(digest(folder/"source_code"/name) == expected, f"Saved source mismatch: {name}")
    checkpoint_path = folder/"trained_model/model.pt"
    require(digest(checkpoint_path) == summary["model_checkpoint_sha256"], "Checkpoint hash mismatch")
    old_summary = read_json(folder/"trained_model/original_training_run_summary.json")
    require(old_summary["source_sha256"]["pinn_model.py"] == digest(folder/"source_code/pinn_model.py"), "Training/inference model source mismatch")
    split_ids = read_json(folder/"trained_model/original_data_splits.json")
    require(all(summary["norad_id"] not in ids for ids in split_ids.values()), "Benchmark object leaked into model splits")
    gp = read_json(folder/"inputs/initial_gp.json")[0]
    raw = read_json(folder/"inputs"/provenance["gp_source"]["raw_file"])
    require(gp in raw and int(gp["NORAD_CAT_ID"]) == summary["norad_id"], "Selected GP differs from raw response")
    reference_source = provenance["reference_source"]
    with zipfile.ZipFile(folder/"inputs"/reference_source["zip_file"]) as archive:
        require(archive.testzip() is None, "ESA ZIP CRC check")
        eof_names = [name for name in archive.namelist() if Path(name).name == reference_source["eof_file"]]
        require(len(eof_names) == 1, "ESA EOF entry missing/ambiguous in ZIP")
        require(archive.read(eof_names[0]) == (folder/"inputs"/reference_source["eof_file"]).read_bytes(), "Extracted EOF differs from ZIP")
    reference = read_npz(folder/"reference/independent_orbit.npz")
    dates, native_r, native_v, dut1, osvs = native_reference(folder/"inputs"/reference_source["eof_file"])
    ix = reference["original_osv_indices"]
    require(np.issubdtype(ix.dtype, np.integer) and np.all(np.diff(ix)>0), "Native OSV indices")
    sample_dates = [utc(value) for value in reference["utc"]]
    require(sample_dates == [dates[i] for i in ix], "Saved times differ from native OSV times")
    require(all(osvs[i].findtext("Quality") == "NOMINAL" for i in ix), "Reference quality")
    epoch = utc(gp["EPOCH"])
    require(ix[0] == next(i for i, value in enumerate(dates) if value >= epoch), "Initial reference timestamp was not first at/after GP epoch")
    seconds = reference["seconds_from_start"]
    close(seconds, [(date-sample_dates[0]).total_seconds() for date in sample_dates], "Elapsed times", atol=0, rtol=0)
    close(np.diff(seconds), summary["step_seconds"], "Fixed time step", atol=0, rtol=0)
    require(len(seconds) == summary["sample_count"] and seconds[0] == 0 and seconds[-1] == summary["horizon_hours"]*3600, "Time horizon/sample count")
    require(sample_dates[0] == utc(summary["start_utc"]) and sample_dates[-1] == utc(summary["end_utc"]), "Summary dates")
    close(reference["position_itrs_km"], native_r[ix], "Native ITRS position", atol=0, rtol=0)
    close(reference["velocity_itrs_km_s"], native_v[ix], "Native ITRS velocity", atol=0, rtol=0)
    rr, rv, flags = transform_reference(sample_dates, native_r[ix], native_v[ix], dut1[ix], folder/"inputs/finals2000A.all")
    close(rr, reference["position_teme_km"], "Reference position transform replay", atol=1e-9, rtol=0)
    close(rv, reference["velocity_teme_km_s"], "Reference velocity transform replay", atol=1e-9, rtol=0)
    frame = summary["frame_transformation"]
    require(digest(folder/"inputs/finals2000A.all") == frame["eop_sha256"], "EOP hash")
    require(frame["polar_motion_status_counts"] == {str(int(flag)):int(sum(flags == flag)) for flag in np.unique(flags)}, "EOP status count")
    predictions = {model:read_npz(folder/"predictions"/(model.lower()+".npz")) for model in ("SGP4", "PINN")}
    stored_metrics = read_json(folder/"evaluation/metrics.json")
    metric_count = 0
    for model, prediction in predictions.items():
        for key in ("utc", "seconds_from_start", "norad_id"):
            require(np.array_equal(prediction[key], reference[key]), f"{model} metadata alignment: {key}")
        metrics, errors = independent_metrics(prediction["position_teme_km"], prediction["velocity_teme_km_s"], rr, rv)
        stored_errors = read_npz(folder/"evaluation"/(model.lower()+"_errors.npz"))
        for key, values in errors.items():
            close(values, stored_errors[key], f"{model} saved errors: {key}")
        for key, value in metrics.items():
            close(value, stored_metrics[model][key], f"{model} metric: {key}")
            close(value, summary["metrics"][model][key], f"{model} summary metric: {key}")
            metric_count += 1
    initial = read_npz(folder/"inputs/common_initial_state.npz")
    for model, prediction in predictions.items():
        close(prediction["position_teme_km"][0], initial["position_teme_km"], f"{model} common initial position")
        close(prediction["velocity_teme_km_s"][0], initial["velocity_teme_km_s"], f"{model} common initial velocity")
    # Direct SGP4 calls avoid the project's propagation wrapper.
    from sgp4 import omm
    from sgp4.api import Satrec, WGS72, jday
    satellite = Satrec()
    fields = dict(gp)
    fields["EPOCH"] = epoch.replace(tzinfo=None).isoformat(timespec="microseconds")
    omm.initialize(satellite, fields, WGS72)
    sgp4_states = []
    for date in sample_dates:
        jd, fraction = jday(date.year, date.month, date.day, date.hour, date.minute, date.second+date.microsecond/1e6)
        code, r, v = satellite.sgp4(jd, fraction)
        require(code == 0, "SGP4 replay status")
        sgp4_states.append([r, v])
    sgp4_states = np.asarray(sgp4_states)
    close(sgp4_states[:, 0], predictions["SGP4"]["position_teme_km"], "SGP4 replay position", atol=1e-7, rtol=0)
    close(sgp4_states[:, 1], predictions["SGP4"]["velocity_teme_km_s"], "SGP4 replay velocity", atol=1e-10, rtol=0)
    # Load the checksummed source/checkpoint; execute the network directly each step.
    import torch
    torch.set_num_threads(summary["hardware"]["torch_threads"])
    spec = importlib.util.spec_from_file_location("saved_pinn_for_verification", folder/"source_code/pinn_model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    network = module.OrbitalPINN(checkpoint["width"], checkpoint["depth"]).double().eval()
    network.load_state_dict(checkpoint["model_state_dict"])
    state = torch.tensor(initial["normalized_state"], dtype=torch.float64)
    neural = [state.numpy().copy()[0]]
    with torch.no_grad():
        for dt in np.diff(seconds):
            require(0 < dt <= checkpoint["segment_seconds"], "PINN trained time interval")
            state = network(state, torch.tensor([[dt/module.TIME_SECONDS]], dtype=torch.float64))
            neural.append(state.numpy().copy()[0])
    neural = np.asarray(neural)
    close(neural[:, :3]*module.LENGTH_KM, predictions["PINN"]["position_teme_km"], "Frozen PINN position replay", atol=1e-9, rtol=0)
    close(neural[:, 3:]*module.SPEED_KM_S, predictions["PINN"]["velocity_teme_km_s"], "Frozen PINN velocity replay", atol=1e-12, rtol=0)
    with (folder/"predictions/all_states.csv").open(newline="", encoding="utf-8") as stream:
        state_rows = list(csv.DictReader(stream))
    require(len(state_rows) == 3*len(seconds), "CSV state row count")
    sources = {**predictions, "ESA_REFERENCE":reference}
    for name, source in sources.items():
        rows = [row for row in state_rows if row["model"] == name]
        require([utc(row["utc"]) for row in rows] == sample_dates, f"{name} CSV timestamps")
        close([[float(row[f"{axis}_km"]) for axis in "xyz"] for row in rows], source["position_teme_km"], f"{name} CSV position")
        close([[float(row[f"v{axis}_km_s"]) for axis in "xyz"] for row in rows], source["velocity_teme_km_s"], f"{name} CSV velocity")
    # A five-point derivative independently checks velocity rotation against r(t).
    dense_ix = np.arange(max(0, int(ix[0])-2), min(len(dates), int(ix[-1])+3))
    dense_dates = [dates[i] for i in dense_ix]
    cadence = np.asarray([(b-a).total_seconds() for a, b in zip(dense_dates, dense_dates[1:])])
    require(len(dense_ix)>=5 and np.all(cadence == cadence[0]), "Native finite-difference cadence")
    dense_r, dense_v, _ = transform_reference(dense_dates, native_r[dense_ix], native_v[dense_ix], dut1[dense_ix], folder/"inputs/finals2000A.all")
    fd = (dense_r[:-4]-8*dense_r[1:-3]+8*dense_r[3:-1]-dense_r[4:])/(12*cadence[0])
    fd_errors_m_s = 1000*np.linalg.norm(fd-dense_v[2:-2], axis=1)
    require(np.max(fd_errors_m_s) < .01, "Transformed velocity disagrees with independent position derivative by >= 0.01 m/s")
    unavailable = read_json(folder/"evaluation/unavailable_classification_metrics.json")
    require(all(unavailable[key] is None for key in ("accuracy", "precision", "recall", "f1_score", "collision_probability")), "Unsupported classification scores must remain unavailable")
    require(all(read_json(folder/"evaluation/verification.json").values()), "Original verification contains failed checks")
    # Do not repair or refresh hashes of any original result: abort if it changed.
    require(manifest_path.read_bytes() == manifest_bytes, "Checksum manifest changed during verification")
    for relative, expected in hashes.items():
        require(digest(folder/relative) == expected, f"Artifact changed during verification: {relative}")
    artifact = {"status":"passed", "verified_at_utc":datetime.now(timezone.utc).isoformat(),
                "run_id":summary["run_id"], "original_manifest_sha256":hashlib.sha256(manifest_bytes).hexdigest(),
                "original_manifest_entries_checked":len(hashes), "numeric_metric_fields_recomputed":metric_count,
                "models_replayed":["SGP4", "PINN frozen checkpoint"], "states_per_model":len(seconds),
                "independent_xml_parse_and_native_time_alignment":True, "input_zip_crc_and_eof_match":True,
                "saved_source_hashes_match_training_and_run_provenance":True,
                "csv_states_match_npz_arrays":True, "all_saved_error_arrays_and_numeric_metrics_match":True,
                "reference_frame_replay_passed":True, "classification_metrics_correctly_unavailable":True,
                "five_point_position_derivative_check":{"native_step_seconds":float(cadence[0]),
                    "interior_samples":len(fd_errors_m_s), "maximum_velocity_difference_m_s":float(max(fd_errors_m_s)),
                    "median_velocity_difference_m_s":float(np.median(fd_errors_m_s)), "pass_tolerance_m_s":.01,
                    "note":"Numerical consistency check using independently differentiated native reference positions, not an external orbit-accuracy bound."},
                "notes":["No model training, network acquisition, website edits, or benchmark value updates.",
                         "Metric formulas and XML parsing are independently implemented; frame replay uses the same Astropy library.",
                         "Checksum validation establishes stored file integrity, not source-data authenticity."]}
    source_target = folder/OWNED[1]
    if Path(__file__).resolve() != source_target.resolve():
        shutil.copy2(__file__, source_target)
    (folder/OWNED[0]).write_text(json.dumps(artifact, indent=2)+"\n", encoding="utf-8")
    for relative in OWNED:
        hashes[relative] = digest(folder/relative)
    manifest_path.write_text(json.dumps(hashes, indent=2)+"\n", encoding="utf-8")
    require(all(digest(folder/name) == value for name, value in hashes.items()), "Final checksum verification failed")
    print(json.dumps(artifact, indent=2))
    return artifact


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_folder", type=Path)
    verify(parser.parse_args().run_folder)
