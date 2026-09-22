"""Verify saved three-model states, weights, split metrics and input checksums."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from fetch_celestrak import parse_epoch, write_json
from hybrid_model import load_corrector, predict_corrected
from orbit_engine import make_satellite, propagate
from orbit_truth import orbit_error_metrics
from pinn_model import LENGTH_KM, SPEED_KM_S, OrbitalPINN, normalize_states, rollout


def verify(folder):
    folder = Path(folder).resolve()
    checksums = json.loads((folder/"file_checksums.json").read_text())
    for relative, expected in checksums.items():
        path = (folder/relative).resolve()
        if not path.is_relative_to(folder) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Artifact checksum mismatch: {relative}")
    summary = json.loads((folder/"summary.json").read_text())
    torch.set_num_threads(2)
    with np.load(folder/"reference/orbit.npz", allow_pickle=False) as data:
        seconds, rr, rv = data["seconds_from_start"], data["position_teme_km"], data["velocity_teme_km_s"]
    if not np.array_equal(seconds, np.arange(181)*60):
        raise ValueError("Expected the locked 181-point native time grid")
    record = json.loads((folder/"inputs/benchmark/initial_gp.json").read_text())[0]
    codes, sr, sv = propagate([make_satellite(record)], parse_epoch(summary["start_utc"]), seconds)
    if np.any(codes): raise ValueError("SGP4 replay failed")
    sr, sv = sr[0], sv[0]
    saved = torch.load(folder/"pinn/model.pt", map_location="cpu", weights_only=True)
    model = OrbitalPINN(saved["width"], saved["depth"]).double().eval()
    model.load_state_dict(saved["model_state_dict"])
    normalized = rollout(model, normalize_states(sr[:1], sv[:1]), seconds)[0]
    corrector, config = load_corrector(folder/"combined/model.pt")
    replay = {"SGP4":{"r":sr, "v":sv},
              "PINN":{"r":normalized[:,:3]*LENGTH_KM, "v":normalized[:,3:]*SPEED_KM_S},
              "COMBINED":predict_corrected(corrector, config, seconds, sr, sv)}
    checks = {"original_checksums_verified":len(checksums), "models":{}}
    for name, p in replay.items():
        with np.load(folder/name.lower()/"predictions.npz", allow_pickle=False) as data:
            r, v = data["position_teme_km"], data["velocity_teme_km_s"]
            max_r, max_v = float(np.max(np.abs(p["r"]-r))), float(np.max(np.abs(p["v"]-v)))
            if max_r > 1e-6 or max_v > 1e-8:
                raise ValueError(f"{name} saved model does not replay")
        if not np.allclose(r[0], sr[0], rtol=0, atol=1e-10) or not np.allclose(v[0], sv[0], rtol=0, atol=1e-12):
            raise ValueError(f"{name} initial condition differs")
        for split, (a,b) in {"train":(0,90), "validation":(90,120), "test":(120,180)}.items():
            recomputed, _ = orbit_error_metrics(r[a:b+1], v[a:b+1], rr[a:b+1], rv[a:b+1])
            original = summary["metrics"][name][split]
            for key, value in recomputed.items():
                if isinstance(value, (int,float,list)) and key != "rtn_axes":
                    if not np.allclose(value, original[key], rtol=1e-11, atol=1e-12):
                        raise ValueError(f"{name}/{split} metric mismatch: {key}")
        # Independently index exactly 121..180, separate from the helper's t0 convention.
        direct_rmse = float(np.sqrt(np.mean(np.sum((r[121:]-rr[121:])**2, axis=1))))
        if abs(direct_rmse-summary["metrics"][name]["test"]["position_vector_rmse_km"]) > 1e-12:
            raise ValueError("Held-out sample indexing error")
        checks["models"][name] = {"checkpoint_position_replay_max_km":max_r, "checkpoint_velocity_replay_max_km_s":max_v,
                                  "all_split_metrics_recomputed":True, "heldout_samples":60}
    for split, lower, upper in (("train",60,5400), ("validation",5460,7200)):
        with np.load(folder/"combined"/(split+"_calibration_samples.npz"), allow_pickle=False) as data:
            if data["seconds"].min() != lower or data["seconds"].max() != upper:
                raise ValueError("Calibration arrays contain unexpected times")
    checks.update(passed=True, no_test_calibration_samples=True, network_weights_reloaded=True,
                  note="Offline artifact replay; no new training, reference downloads, or provider calls.")
    write_json(folder/"comparison/replay_verification.json", checks)
    print("Verified checksums, all three saved predictions, neural weights and every split metric.", flush=True)
    return checks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    verify(parser.parse_args().folder)
