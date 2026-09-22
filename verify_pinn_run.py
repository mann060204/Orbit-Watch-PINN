"""Reload a saved checkpoint, replay predictions, and independently check saved metrics."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from fetch_celestrak import utc_now, utc_text, write_json
from pinn_model import OrbitalPINN, LENGTH_KM, SPEED_KM_S, rollout

ROOT = Path(__file__).resolve().parent


def verify(folder):
    folder = Path(folder).resolve()
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    checks = {}
    for filename, expected in summary["source_sha256"].items():
        if Path(filename).name != filename:
            raise ValueError("Invalid source filename")
        actual = hashlib.sha256((ROOT / filename).read_bytes()).hexdigest()
        checks[f"source_hash:{filename}"] = actual == expected
    torch.set_num_threads(2)
    checkpoint = torch.load(folder / "model.pt", map_location="cpu", weights_only=True)
    model = OrbitalPINN(checkpoint["width"], checkpoint["depth"]).double().eval()
    model.load_state_dict(checkpoint["model_state_dict"])
    with np.load(folder / "initial_conditions.npz", allow_pickle=False) as initial:
        ids, x0 = initial["norad_ids"], initial["normalized_state"]
    with np.load(folder / "pinn_trajectories.npz", allow_pickle=False) as trajectories:
        seconds = trajectories["seconds_from_start"]
        saved_r, saved_v = trajectories["position_teme_km"], trajectories["velocity_teme_km_s"]
        checks["trajectory_ids_match"] = np.array_equal(ids, trajectories["norad_ids"])
    replay = rollout(model, x0, seconds)
    max_position_difference = float(np.max(np.abs(replay[:, :, :3]*LENGTH_KM - saved_r)))
    max_velocity_difference = float(np.max(np.abs(replay[:, :, 3:]*SPEED_KM_S - saved_v)))
    checks["checkpoint_replays_all_positions"] = max_position_difference < 1e-8
    checks["checkpoint_replays_all_velocities"] = max_velocity_difference < 1e-11
    splits = json.loads((folder / "data_splits.json").read_text(encoding="utf-8"))
    train, validation, test = (set(splits[name]) for name in ("train", "validation", "test"))
    checks["splits_disjoint_and_complete"] = not (train & validation or train & test or validation & test) and train | validation | test == set(ids.tolist())
    with np.load(folder / "sgp4_reference_trajectories.npz", allow_pickle=False) as reference:
        errors = np.linalg.norm(saved_r - reference["position_teme_km"], axis=-1)
        checks["reference_states_valid"] = bool((reference["sgp4_errors"] == 0).all())
    test_indices = np.array([i for i, value in enumerate(ids) if int(value) in test])
    test_rmse = float(np.sqrt(np.mean(errors[test_indices, 1:]**2)))
    reported_rmse = summary["metrics"]["trajectory_agreement_vs_sgp4"]["test"]["position_vector_rmse_km"]
    checks["heldout_rmse_independently_recomputed"] = abs(test_rmse - reported_rmse) < 1e-10
    events = json.loads((folder / "pinn_conjunctions.json").read_text(encoding="utf-8"))
    reference_events = json.loads((folder / "sgp4_reference_events.json").read_text(encoding="utf-8"))
    pairs = lambda rows: {tuple(sorted((int(row["object1_id"]), int(row["object2_id"])))) for row in rows}
    predicted, expected = pairs(events), pairs(reference_events)
    with (folder / "shared_orbit_pairs.csv").open(encoding="utf-8", newline="") as stream:
        shared = pairs(csv.DictReader(stream))
    universe = len(ids)*(len(ids)-1)//2 - len(shared)
    confusion = {"true_positive":len(predicted & expected), "false_positive":len(predicted - expected),
                 "false_negative":len(expected - predicted), "true_negative":universe-len(predicted | expected)}
    checks["confusion_matrix_independently_recomputed"] = confusion == summary["metrics"]["sgp4_reference_pair_classification"]["confusion_matrix"]
    checks["event_count_matches"] = len(events) == summary["pinn_close_approach_events"]
    checks["saved_model_is_selected_checkpoint"] = checkpoint["best_step"] == summary["training"]["best_step"]
    if summary["metrics"]["collision_probability"]["status"] == "unavailable":
        checks["missing_covariances_produce_no_probability"] = all(event["collision_probability"] is None for event in events)
    outcome = {"verified_at_utc":utc_text(utc_now()), "run_id":folder.name, "passed":all(checks.values()),
               "checks":checks, "replayed_future_states":len(ids)*(len(seconds)-1),
               "maximum_checkpoint_replay_position_difference_km":max_position_difference,
               "maximum_checkpoint_replay_velocity_difference_km_s":max_velocity_difference,
               "recomputed_test_position_rmse_km":test_rmse,
               "meaning":"Artifact consistency and reproducibility checks; not observational validation."}
    write_json(folder / "artifact_verification.json", outcome)
    print(json.dumps(outcome, indent=2))
    if not outcome["passed"]:
        raise SystemExit("Saved-run verification failed")
    return outcome


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", nargs="?", type=Path)
    args = parser.parse_args()
    latest = ROOT / "pinn_results" / "latest.json"
    folder = args.folder or ROOT / "pinn_results" / json.loads(latest.read_text())["run_id"]
    verify(folder)
