"""Static files for an offline comparison; no dashboard or website publication."""
import os

import numpy as np


def make_comparison_report(folder, summary, history, time_rows, position_errors, event_rows):
    os.environ.setdefault("MPLCONFIGDIR", str(folder/"matplotlib_cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = summary["metrics"]
    pair = m["pinn_vs_sgp4_all_pairs"]
    test_pairs = m["pinn_vs_sgp4_test_only_pairs"]
    test = m["pinn_vs_sgp4_trajectory"]["test"]
    timing = m["model_comparison"]
    training = m["pinn_training"]
    def fmt(value, digits=6):
        return "Unavailable" if value is None else f"{value:.{digits}g}" if isinstance(value, (int, float)) else str(value)
    fig, ax = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for split, color in (("train", "#15889a"), ("validation", "#7960ab"), ("test", "#d77e33")):
        selected = [row for row in time_rows if row["split"] == split]
        ax[0, 0].plot([row["seconds_from_start"]/60 for row in selected],
                      [row["position_vector_rmse_km"] for row in selected], label=split, color=color)
    ax[0, 0].set(title="Position disagreement with SGP4", xlabel="Minutes from common start", ylabel="Position vector RMSE (km)")
    ax[0, 0].legend()
    for field, label, color in (("training_physics_loss", "training", "#15889a"),
                                 ("validation_physics_loss", "validation", "#d77e33")):
        ax[0, 1].semilogy([row["step"] for row in history], [row[field] for row in history], label=label, color=color)
    ax[0, 1].set(title="Actual PINN training", xlabel="Optimizer steps", ylabel="Scaled physics residual MSE")
    ax[0, 1].legend()
    cm = pair["confusion_matrix"]
    matrix = np.array([[cm["true_negative"], cm["false_positive"]], [cm["false_negative"], cm["true_positive"]]])
    ax[1, 0].imshow(np.log1p(matrix), cmap="Blues")
    for (i, j), value in np.ndenumerate(matrix):
        ax[1, 0].text(j, i, f"{value:,}", ha="center", va="center", color="white" if value>matrix.max()/2 else "black", fontsize=13)
    ax[1, 0].set(title="All-object pair agreement (SGP4 reference)", xticks=[0, 1], xticklabels=["PINN negative", "PINN positive"],
                  yticks=[0, 1], yticklabels=["SGP4 negative", "SGP4 positive"])
    models = ["SGP4", "PINN"]
    values = [timing[name]["propagation_seconds_median"] for name in models]
    bars = ax[1, 1].bar(models, values, color=["#15889a", "#d77e33"])
    ax[1, 1].bar_label(bars, labels=[f"{value:.4f} s" for value in values], padding=4)
    ax[1, 1].set(title="Measured propagation time (training excluded)", ylabel="Median wall-clock seconds")
    ax[1, 1].set_ylim(0, max(values)*1.25)
    fig.suptitle("SGP4 vs PINN | same CelesTrak snapshot and forecast window | model agreement, not observations", fontsize=13)
    fig.savefig(folder/"figures/comparison_summary.png", dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    final_errors = np.sort(position_errors[:, -1])
    ax[0].plot(final_errors, np.arange(1, len(final_errors)+1)/len(final_errors), color="#15889a")
    ax[0].set(title="Final position disagreement: all objects", xlabel="Final PINN–SGP4 distance (km)", ylabel="Cumulative fraction")
    matched = [row for row in event_rows if row["status"] == "matched"]
    if matched:
        ax[1].scatter([row["sgp4_miss_distance_km"] for row in matched], [row["pinn_miss_distance_km"] for row in matched], color="#d77e33")
        bound = summary["config"]["threshold_km"]
        ax[1].plot([0, bound], [0, bound], "--", color="#999999")
    else:
        ax[1].text(.5, .5, "No matched encounters", transform=ax[1].transAxes, ha="center")
    ax[1].set(title="One-to-one matched close approaches", xlabel="SGP4 miss distance (km)", ylabel="PINN miss distance (km)")
    fig.savefig(folder/"figures/trajectory_and_encounter_errors.png", dpi=170)
    plt.close(fig)
    rows = []
    for label, key in (("Close approaches", "close_approach_events"), ("Closest approach (km)", "closest_approach_km"),
                       ("Median propagation time (s)", "propagation_seconds_median"), ("Screening routine time (s)", "screening_seconds"),
                       ("Training time (s)", "training_seconds"), ("Valid future state samples", "valid_future_states")):
        rows.append(f"| {label} | {fmt(timing['SGP4'][key])} | {fmt(timing['PINN'][key])} |")
    classification_rows = [f"| {label} | {fmt(pair[label])} | {fmt(test_pairs[label])} |" for label in
                           ("accuracy", "precision", "recall", "f1_score", "specificity", "balanced_accuracy",
                            "negative_predictive_value", "false_positive_rate", "false_negative_rate", "matthews_correlation_coefficient")]
    report = f"""# SGP4 versus PINN — saved comparison

Run: `{summary['run_id']}`

Both models use the same {summary['modeled_objects']:,} eligible CelesTrak objects, the same initial
states, and the same UTC forecast window: **{summary['window_start_utc']} to {summary['window_end_utc']}**.
The step is {summary['config']['step_seconds']:g} seconds and the close-approach threshold is
{summary['config']['threshold_km']:g} km. Results are saved locally; no website results or pointers were updated.

## Side-by-side measured results

| Quantity | SGP4 | PINN |
| --- | ---: | ---: |
"""+"\n".join(rows)+f"""

SGP4 is an analytical propagator and requires no neural training. PINN training actually executed
{training['optimizer_steps_executed']:,} optimizer steps. Median propagation timing uses repeated warmed
runs in alternating order, excludes disk I/O and training, and includes array computation/device transfers.
The PINN/SGP4 propagation-time ratio is **{timing['pinn_over_sgp4_propagation_time_ratio']:.3f}**;
values above 1 mean PINN was slower in this experiment. This is one machine/session, not a universal benchmark.
Screening timings cover the existing routines; SGP4 additionally creates encounter distance curves.

## Accuracy, precision, recall and other agreement scores

These are agreement with SGP4 pair flags, **not independently measured collision accuracy**.
SGP4 is not awarded 100% accuracy by comparing it with itself. All-object pair metrics include
training and validation objects; the test-only column requires both members to be held out.

| Metric | All eligible pairs | Test-only pairs |
| --- | ---: | ---: |
"""+"\n".join(classification_rows)+f"""

All-pair confusion counts: {pair['confusion_matrix']}. Pair universe: {pair['pair_universe']:,}.
Test-only positive reference pairs: {test_pairs['reference_positive_pairs']}.
Undefined denominators remain null. Accuracy is dominated by negative pairs; inspect precision,
recall and the positive counts before interpreting it. The exact values are saved in JSON and CSV.

## Trajectory and encounter differences

- Held-out position vector RMSE: **{test['position_vector_rmse_km']:.6g} km**.
- Held-out position vector MAE: {test['position_vector_mae_km']:.6g} km.
- Held-out 95th percentile / maximum position error: {test['position_vector_p95_km']:.6g} / {test['position_vector_max_km']:.6g} km.
- Held-out final-time position vector RMSE: {test['final_position_vector_rmse_km']:.6g} km.
- Held-out velocity vector RMSE: {test['velocity_vector_rmse_km_s']:.6g} km/s.
- Matched / PINN-only / SGP4-only encounter episodes: {m['one_to_one_event_agreement']['matched_events']} / {m['one_to_one_event_agreement']['pinn_only_events']} / {m['one_to_one_event_agreement']['sgp4_only_events']}.
- Matched miss-distance MAE: {fmt(m['one_to_one_event_agreement']['miss_distance_difference_km_mae'])} km.
- Matched closest-approach time MAE: {fmt(m['one_to_one_event_agreement']['tca_difference_seconds_mae'])} seconds.

Trajectory metrics exclude the identical initial state. Radial/transverse/normal errors use the
instantaneous SGP4 orbital basis. Per-object, per-time, per-axis and split-level errors are saved.
Encounter matching is one-to-one within {m['one_to_one_event_agreement']['matching_tolerance_seconds']:g} seconds
for the same catalog pair. Both models' unmatched events are retained. Pair classification means any
flagged approach within the whole window; it is different from episode matching.

## Data and actual training

Input records: {summary['input_objects']:,}; excluded: {summary['excluded_objects']:,};
indistinguishable modeled-orbit pairs separated: {summary['shared_orbit_pairs']}.
The selected four groups are a subset of the catalog. Download/epoch times and cache use are in
`{summary['input_snapshot']}/manifest.json`. Cached GP sets are actual downloaded data, not new observations.

Device: {summary['hardware']['selected_device']}; {summary['hardware']['torch_threads']} CPU threads;
CPU: {summary['hardware'].get('cpu', 'see hardware.json')}.
Object split counts: {summary['split_counts']}.
Trainable parameters: {training['parameter_count']:,}; best validation step: {training['best_step']:,};
peak sampled training-process memory: {training['peak_sampled_resident_memory_mb']:.1f} MB.

PINN is the existing three-layer tanh correction to a Taylor trial solution with hard initial conditions.
It is trained using automatic-differentiation residuals for central gravity plus J2. Initial Cartesian
states come from SGP4 applied to the real GP records. **Zero future SGP4 training targets** are used.
Checkpoint selection uses validation physics residuals. Future SGP4 tracks are computed only after
training. The trained local flow map is iterated across the forecast window.

## What this comparison can and cannot establish

The outputs and training/evaluation measurements are real executions on downloaded data. The orbital
trajectories are model forecasts, not observed future positions. No independent measured trajectories
or collision outcomes exist in this input, so absolute observational accuracy for **both** SGP4 and
PINN remains unavailable. ROC AUC, PR AUC, Brier score and collision probability are also unavailable;
covariances, justified object radii and calibrated scores were not invented.

Physics diagnostics evaluate central+J2 energy/axial-momentum drift and finite-difference residuals
for both models. They include time-discretization error. SGP4 models different perturbations, so those
diagnostics do not establish which model is more accurate in the real world. PINN omits drag, J3/J4,
third bodies and maneuvers, and treats TEME as quasi-inertial. Accumulated position errors matter for
kilometer-scale close-approach screening. Shared-orbit entries cannot be independently resolved.
Swept-chord screening with assumed acceleration padding and local minima is not a proof of exhaustive detection.

## Files

- `sgp4/`: full trajectories, error codes, encounters, and screening audits.
- `pinn/`: trained model weights, trajectories, encounters, training logs/proof, and screening audits.
- `evaluation/all_metrics.csv`, `metrics.json`: every reported evaluation and explicit unavailable values.
- `evaluation/model_comparison.csv`: side-by-side model measurements.
- `evaluation/per_object_errors.csv`, `errors_by_forecast_time.csv`, `trajectory_differences.npz`: regression details.
- `evaluation/pair_agreement.csv`, `pair_universe.json`, `event_matches.csv`: pair and one-to-one event comparisons.
- `evaluation/propagation_benchmarks.csv`, `artifact_verification.json`: actual timings and artifact checks.
- `figures/`: static comparison charts. `source_code/`: exact source and dependency snapshots.
- `initial_conditions.npz`, `data_splits.json`, `eligible_objects.json`, `excluded_objects.json`: common inputs and coverage.
- `summary.json`, `config.json`, `hardware.json`, `file_checksums.json`, `run.log`: provenance and reproducibility.

## Reproduce

From the main project with the pinned dependencies installed:

```powershell
.\\run_comparison.ps1 --snapshot "{(folder / summary['input_snapshot']).as_posix()}" --start "{summary['window_start_utc']}" --steps {training['optimizer_steps_executed']} --hours {summary['config']['horizon_hours']:g} --segment-seconds {summary['config']['step_seconds']:g} --threshold-km {summary['config']['threshold_km']:g}
```

References: [CelesTrak GP formats and SGP4 conventions](https://celestrak.org/NORAD/documentation/gp-data-formats.php),
[NASA close-approach risk assessment](https://www.nasa.gov/cara/step-2-close-approach-risk-assessment/).
"""
    (folder/"comparison_report.md").write_text(report, encoding="utf-8")
