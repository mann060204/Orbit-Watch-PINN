"""Run SGP4, screen close approaches, and save a reproducible results folder."""
from __future__ import annotations

import argparse
import csv
from datetime import timedelta
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import shutil
import time
import traceback

import numpy as np

from evaluation import CLASSIFICATION_NAMES, classification_metrics
from fetch_celestrak import parse_epoch, utc_now, utc_text, validate_records, write_json
from orbit_engine import AnalysisConfig, prepare_catalog, propagate, reference_validation, screen

ROOT = Path(__file__).resolve().parent
LIMITATIONS = [
    "Close approaches within a chosen distance are screening flags, not confirmed collisions or probabilities.",
    "CelesTrak mean elements are estimates; there are no position covariances, hard-body radii, maneuvers, or collision labels here.",
    "Only objects in the selected input catalog are screened; untracked debris and omitted groups are outside coverage.",
    "Staleness is assessed at analysis start. Epoch ages at encounter time are also reported.",
    "Broad-phase swept chords use an assumed acceleration bound of A km/s^2 with per-object padding A*dt^2/8. This is not uncertainty padding or a proof of complete SGP4 detection.",
    "Bounded time refinement assumes a smooth local distance minimum in each short interval. Shorter-step convergence studies are needed for coverage assessment.",
    "One closest minimum is reported per contiguous candidate episode. Co-orbiting/docked objects can generate persistent small separations and need interpretation.",
    "Pairs with exactly identical modeled elements are excluded from the encounter register and saved in shared_orbit_pairs.csv; their physical separation cannot be inferred from these elements.",
    "Numerical agreement with a published SGP4 case measures implementation consistency, not orbit-prediction accuracy.",
]


def read_snapshot(data_dir=ROOT / "data"):
    data_dir = Path(data_dir).resolve()
    latest = json.loads((data_dir / "latest.json").read_text(encoding="utf-8"))
    snapshot = (data_dir / latest["snapshot"]).resolve()
    if not snapshot.is_relative_to(data_dir / "snapshots"):
        raise ValueError("Snapshot path is outside the data/snapshots folder")
    records = validate_records(json.loads((snapshot / "orbital_data.json").read_text(encoding="utf-8")))
    return snapshot, latest, records


def write_csv(path, rows, fields=None):
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value
                             for key, value in row.items() if key in fields})


def make_figures(run_dir, events, rows, start, summary):
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ages = [(start - parse_epoch(row["EPOCH"])).total_seconds() / 3600 for row in rows]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    axes[0].hist(ages, bins=35, color="#167c9b")
    axes[0].axvline(summary["config"]["max_epoch_age_hours"], color="#d55c32", linestyle="--", label="Age limit")
    axes[0].set(xlabel="Element age at start (hours)", ylabel="Objects", title="Input freshness")
    axes[0].legend()
    if events:
        axes[1].hist([e["miss_distance_km"] for e in events], bins=25, color="#dd9143")
        axes[2].scatter([e["tca_offset_seconds"] / 3600 for e in events],
                        [e["miss_distance_km"] for e in events], s=12, color="#167c9b", alpha=0.65)
    else:
        for axis in axes[1:]:
            axis.text(0.5, 0.5, "No close approaches inside threshold", transform=axis.transAxes, ha="center", wrap=True)
    axes[1].set(xlabel="Miss distance (km)", ylabel="Events", title="Screened close approaches")
    axes[2].set(xlabel="Hours from analysis start", ylabel="Miss distance (km)", title="Encounter timeline")
    fig.suptitle("SGP4 research screening | distances do not express collision probability", fontsize=13)
    (run_dir / "figures").mkdir()
    fig.savefig(run_dir / "figures" / "screening_summary.png", dpi=180)
    plt.close(fig)


def run_analysis(config=None, data_dir=ROOT / "data", results_dir=ROOT / "results", start=None,
                 labels_path=None, progress=lambda *args: None):
    config = (config or AnalysisConfig()).validate()
    start = start or utc_now()
    results_dir = Path(results_dir)
    snapshot, input_manifest, records = read_snapshot(data_dir)
    run_id = utc_now().strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir = results_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    began = time.perf_counter()
    log = []
    def report(percent, message):
        log.append(f"{utc_text(utc_now())} | {percent:.1f}% | {message}")
        progress(percent, message)
    write_json(run_dir / "status.json", {"status": "running", "run_id": run_id})
    try:
        report(2, "Loading orbital elements and initializing SGP4")
        satellites, rows, initialization_errors = prepare_catalog(records)
        seconds = np.arange(0, config.horizon_hours * 3600, config.step_seconds, dtype=float)
        seconds = np.append(seconds, config.horizon_hours * 3600)
        estimate = len(rows) * len(seconds) * 60
        if estimate > 850 * 1024**2:
            raise ValueError("This time grid needs too much memory. Increase the step or shorten the horizon.")
        write_json(run_dir / "config.json", {**config.to_dict(), "window_start_utc": utc_text(start)})
        shutil.copytree(snapshot, run_dir / "input_snapshot")
        report(8, f"Propagating {len(rows):,} objects at {len(seconds):,} times")
        errors, positions, velocities = propagate(satellites, start, seconds)
        propagation_seconds = time.perf_counter() - began
        report(18, "Screening eligible trajectories for close approaches")
        events, audit, exclusions, refinement_errors, stats = screen(
            satellites, rows, positions, errors, start, seconds, config, report)
        screening_seconds = time.perf_counter() - began - propagation_seconds
        validation = reference_validation()
        if not validation["passed"]:
            raise ValueError("SGP4 numerical reference validation failed")
        end = start + timedelta(seconds=float(seconds[-1]))
        finite = np.isfinite(positions).all(axis=2) & np.isfinite(velocities).all(axis=2)
        success = (errors == 0) & finite
        summary = {
            "schema_version": 1, "run_id": run_id, "created_at_utc": utc_text(utc_now()),
            "window_start_utc": utc_text(start), "window_end_utc": utc_text(end),
            "config": config.to_dict(), "input_objects": len(records), "initialized_objects": len(rows),
            "input_manifest": input_manifest, "frame": "TEME", "position_unit": "km", "velocity_unit": "km/s",
            "gravity_model": "WGS72", "screening": stats,
            "propagation": {"time_samples_per_object": len(seconds), "attempted_states": int(errors.size),
                            "valid_states": int(success.sum()), "failed_states": int((~success).sum()),
                            "success_rate": float(success.mean()), "initialization_failures": len(initialization_errors)},
            "closest_approach_km": events[0]["miss_distance_km"] if events else None,
            "collision_probability": None, "classification": None, "validation": validation,
            "limitations": LIMITATIONS,
            "software": {"python": platform.python_version(), **{name: version(name) for name in ["numpy", "scipy", "sgp4", "Flask", "matplotlib"]}},
            "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                              for name in ["orbit_engine.py", "run_analysis.py", "evaluation.py", "fetch_celestrak.py"]},
        }
        labels = json.loads(Path(labels_path).read_text(encoding="utf-8")) if labels_path else None
        if labels is not None:
            write_json(run_dir / "reference_labels.json", labels)
        summary["classification"] = classification_metrics(events, summary, labels)
        report(86, "Saving trajectories, encounter details, metrics, and reference checks")
        np.savez_compressed(run_dir / "trajectories.npz", norad_ids=np.array([int(row["NORAD_CAT_ID"]) for row in rows]),
                            seconds_from_start=seconds, window_start_utc=np.array(utc_text(start)),
                            position_teme_km=positions, velocity_teme_km_s=velocities, sgp4_error_codes=errors)
        write_json(run_dir / "conjunctions.json", events)
        fields = ["event_id", "object1_id", "object2_id", "object1_name", "object2_name", "tca_utc",
                  "miss_distance_km", "relative_speed_km_s", "object1_epoch_age_hours_at_tca",
                  "object2_epoch_age_hours_at_tca", "stale_at_start", "at_window_boundary", "refinement_failed", "collision_probability"]
        write_csv(run_dir / "conjunctions.csv", events, fields)
        write_csv(run_dir / "candidate_audit.csv", audit, ["object1_id", "object2_id", "bracket_start_utc",
                  "bracket_end_utc", "minimum_distance_km", "refinement_failed"])
        write_json(run_dir / "excluded_objects.json", initialization_errors + exclusions)
        write_csv(run_dir / "shared_orbit_pairs.csv", stats["shared_orbit_pairs"],
                  ["object1_id", "object2_id", "object1_name", "object2_name", "reason"])
        write_csv(run_dir / "excluded_objects.csv", initialization_errors + exclusions,
                  ["norad_id", "reason", "epoch_age_hours", "error_codes"])
        write_json(run_dir / "refinement_errors.json", refinement_errors)
        object_rows = []
        eligible_id_set = set(stats["eligible_ids"])
        propagation_error_rows = []
        for index, row in enumerate(rows):
            object_rows.append({"norad_id": int(row["NORAD_CAT_ID"]), "name": row.get("OBJECT_NAME", ""),
                                "epoch_utc": row["EPOCH"], "source_groups": row.get("SOURCE_GROUPS", []),
                                "epoch_age_hours_at_start": (start - parse_epoch(row["EPOCH"])).total_seconds() / 3600,
                                "screened": int(row["NORAD_CAT_ID"]) in eligible_id_set,
                                "valid_states": int(success[index].sum()),
                                **{key: float(value) if np.isfinite(value) else None for key, value in
                                   zip(["x_km", "y_km", "z_km", "vx_km_s", "vy_km_s", "vz_km_s"],
                                       [*positions[index, 0], *velocities[index, 0]])}})
            bad = np.flatnonzero(~success[index])
            if len(bad):
                propagation_error_rows.append({"norad_id": int(row["NORAD_CAT_ID"]), "failed_states": len(bad),
                                               "first_failure_offset_seconds": float(seconds[bad[0]]),
                                               "error_codes": sorted(set(int(e) for e in errors[index, bad]))})
        write_csv(run_dir / "objects_and_initial_states.csv", object_rows)
        write_csv(run_dir / "propagation_errors.csv", propagation_error_rows,
                  ["norad_id", "failed_states", "first_failure_offset_seconds", "error_codes"])
        write_json(run_dir / "labels_template.json", {
            "target": "close_approach_within_threshold", "threshold_km": config.threshold_km,
            "window_start_utc": utc_text(start), "window_end_utc": utc_text(end),
            "reference_source": "", "pairs": [],
        })
        report(94, "Generating summary figures and the run report")
        make_figures(run_dir, events, rows, start, summary)
        summary["timing_seconds"] = {"propagation_and_loading": propagation_seconds,
                                     "screening_and_refinement": screening_seconds,
                                     "total": time.perf_counter() - began}
        metrics = {"classification": summary["classification"], "propagation": summary["propagation"],
                   "numerical_validation": validation, "screening": stats, "timing_seconds": summary["timing_seconds"]}
        write_json(run_dir / "metrics.json", metrics)
        metric_rows = [{"metric": name, "value": summary["classification"][name],
                        "status": summary["classification"]["status"], "meaning": "Reference-labeled close-approach classification"}
                       for name in (*CLASSIFICATION_NAMES, "roc_auc", "pr_auc", "brier_score")]
        metric_rows += [{"metric": "propagation_success_rate", "value": summary["propagation"]["success_rate"],
                         "status": "measured", "meaning": "Fraction of SGP4 states with no error; not prediction accuracy"},
                        {"metric": "reference_position_error_km", "value": validation["position_vector_error_km"],
                         "status": "measured", "meaning": "Agreement with one published numerical test case"}]
        write_csv(run_dir / "metrics.csv", metric_rows)
        write_json(run_dir / "validation.json", validation)
        write_json(run_dir / "summary.json", summary)
        accuracy_text = "Not available: independent labels are required." if labels is None else json.dumps(summary["classification"], indent=2)
        report_text = f"""# SGP4 close-approach screening report

Run: {run_id}
Window (UTC): {utc_text(start)} to {utc_text(end)}
Input catalog: {len(records):,} objects. Screened: {stats['screened_objects']:,} objects / {stats['possible_pairs']:,} unique pairs.
Distance threshold: {config.threshold_km} km. Time step: {config.step_seconds} seconds.
Close-approach episodes inside threshold: {len(events):,}.
Closest reported approach: {summary['closest_approach_km']} km.
Propagation succeeded for {summary['propagation']['valid_states']:,} / {errors.size:,} states.
Refinement failures: {stats['refinement_failures']}.

## Accuracy, precision, recall, and related metrics

{accuracy_text}

All metrics, including unavailable values as null, are in metrics.json and metrics.csv.
Propagation success rate is not prediction accuracy. Numerical reference agreement is not observational accuracy.
Collision probability is unavailable because covariance and combined object size are missing.

## Files

- input_snapshot/: exact input records and original source metadata.
- config.json, summary.json: reproducible settings, time window, counts, software versions, and source hashes.
- trajectories.npz: ALL initialized objects at ALL time samples, including position and velocity in TEME, time offsets, catalog IDs, and SGP4 error codes. Invalid vectors are NaN; never use them for screening.
- objects_and_initial_states.csv: object metadata, eligibility, and state at analysis start.
- conjunctions.csv / conjunctions.json: refined encounters; JSON also includes distance curves and the two TEME positions at closest approach.
- candidate_audit.csv: every candidate episode and its refined minimum, including those outside the reporting threshold.
- excluded_objects.csv / .json, propagation_errors.csv, refinement_errors.json: quality and failure records.
- shared_orbit_pairs.csv: catalog pairs with identical modeled orbital elements, excluded from the encounter register because their independent separation cannot be resolved.
- metrics.csv / .json, validation.json: evaluation status, measured operational metrics, and the numerical reference check.
- labels_template.json: schema for independent close-approach labels. Include negative pairs as well as positive pairs to evaluate false alarms and missed events. Do not label from this run's own predictions.
- figures/screening_summary.png: freshness, miss-distance distribution, and encounter timeline.
- run.log, status.json: progress log and completion status.

## Method and limits

SGP4 with WGS72, direct OMM initialization, split Julian dates, and TEME coordinates in kilometers and kilometers per second.
Spatial search uses swept interval chords, with per-orbit padding A*dt^2/8 (A={config.acceleration_pad_km_s2} km/s^2).
Candidate intervals are refined with direct SGP4 and a bounded distance minimizer (1 ms numerical time tolerance; not physical time accuracy).

""" + "\n".join(f"- {item}" for item in LIMITATIONS)
        (run_dir / "report.md").write_text(report_text + "\n", encoding="utf-8")
        write_json(run_dir / "status.json", {"status": "complete", "run_id": run_id})
        write_json(results_dir / "latest.json", {"run_id": run_id})
        report(100, f"Complete: {len(events):,} close approaches saved in results/{run_id}")
        return run_dir, summary
    except Exception as exc:
        write_json(run_dir / "status.json", {"status": "failed", "run_id": run_id, "error": str(exc)})
        (run_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        (run_dir / "run.log").write_text("\n".join(log) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=float, default=24)
    parser.add_argument("--step", type=float, default=60)
    parser.add_argument("--threshold-km", type=float, default=5)
    parser.add_argument("--max-age-hours", type=float, default=72)
    parser.add_argument("--include-stale", action="store_true")
    parser.add_argument("--start", help="ISO UTC start, defaults to now")
    parser.add_argument("--labels", type=Path, help="Independent pair labels using the saved JSON template")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    config = AnalysisConfig(args.hours, args.step, args.threshold_km, args.max_age_hours, args.include_stale)
    try:
        folder, summary = run_analysis(config, args.data_dir, args.results_dir,
                                       parse_epoch(args.start) if args.start else None, args.labels,
                                       lambda percent, message: print(f"{percent:5.1f}% {message}", flush=True))
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f"Analysis failed: {exc}\n")
    print(f"Results: {folder.resolve()}\nClose approaches: {summary['screening']['conjunction_events']}")


if __name__ == "__main__":
    main()
