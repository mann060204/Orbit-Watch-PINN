"""Disk-only reports and scientific figures for an executed three-model benchmark.

This module consumes supplied predictions, errors, metrics, and training histories.
It neither fits a model nor creates substitute orbit observations or scores.
"""
from pathlib import Path
import math

import numpy as np


MODEL_NAMES = ("SGP4", "PINN", "COMBINED")
MODEL_LABELS = {"SGP4": "SGP4", "PINN": "PINN", "COMBINED": "SGP4 + neural correction"}
MODEL_COLORS = {"SGP4": "#C65327", "PINN": "#196DB0", "COMBINED": "#7A3E9D"}
MODEL_STYLES = {"SGP4": "--", "PINN": "-.", "COMBINED": "-"}


def make_three_model_report(folder, seconds, truth_r, truth_v, predictions,
                            errors, metrics, meta, histories):
    """Save separate model reports/figures and a comparison; return relative paths.

    Input units are TEME km and km/s. ``metrics[name][split]`` contains the
    already-computed regression metrics, and ``errors`` covers the whole window.
    The held-out set is strictly after ``validation_end_seconds`` and at or before
    ``test_end_seconds``. Its preceding boundary is not counted as a test sample.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = Path(folder)
    seconds = np.asarray(seconds, dtype=float)
    if seconds.ndim != 1 or len(seconds) < 4 or not np.isfinite(seconds).all():
        raise ValueError("At least four finite time offsets are required")
    if seconds[0] != 0 or np.any(np.diff(seconds) <= 0):
        raise ValueError("Time offsets must start at zero and increase strictly")
    count = len(seconds)

    def finite_array(value, shape, name):
        array = np.asarray(value, dtype=float)
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError(f"{name} must contain finite values with shape {shape}")
        return array

    truth_r = finite_array(truth_r, (count, 3), "Reference position")
    truth_v = finite_array(truth_v, (count, 3), "Reference velocity")
    states, checked_errors = {}, {}
    for name in MODEL_NAMES:
        states[name] = {key: finite_array(predictions[name][key], (count, 3), f"{name} {key}")
                        for key in ("r", "v")}
        checked_errors[name] = {
            key: finite_array(errors[name][key],
                              (count,) if "norm" in key else (count, 3), f"{name} {key}")
            for key in ("position_norm_km", "velocity_norm_km_s", "position_xyz_km",
                        "velocity_xyz_km_s", "position_rtn_km", "velocity_rtn_km_s")
        }
        for split_name in ("train", "validation", "test"):
            if not isinstance(metrics[name][split_name], dict):
                raise ValueError(f"Missing {name} {split_name} metrics")

    split = meta.get("split", {})
    train_end = float(split.get("train_end_seconds", 5400))
    validation_end = float(split.get("validation_end_seconds", 7200))
    test_end = float(split.get("test_end_seconds", seconds[-1]))
    if not (0 < train_end < validation_end < test_end <= seconds[-1]):
        raise ValueError("Invalid chronological split boundaries")
    test_mask = (seconds > validation_end) & (seconds <= test_end)
    train_mask = (seconds > 0) & (seconds <= train_end)
    validation_mask = (seconds > train_end) & (seconds <= validation_end)
    split_masks = {"train": train_mask, "validation": validation_mask, "test": test_mask}
    if not all(mask.any() for mask in split_masks.values()):
        raise ValueError("Every evaluation split requires at least one sample")
    for name in MODEL_NAMES:
        for split_name, mask in split_masks.items():
            reported = metrics[name][split_name].get("future_samples")
            if reported is not None and int(reported) != int(mask.sum()):
                raise ValueError(f"{name} {split_name} metric sample count disagrees with the time mask")

    minutes = seconds / 60
    test_count = int(test_mask.sum())
    test_indices = np.flatnonzero(test_mask)
    object_name = str(meta.get("object_name", "Reference satellite"))
    reference_label = str(meta.get("reference_label", "ESA GNSS restituted orbit (RESORB)"))
    outputs = []
    for name in ("sgp4", "pinn", "combined", "comparison"):
        (folder / name / "figures").mkdir(parents=True, exist_ok=True)

    def finish(fig, directory, stem, title, note=""):
        fig.suptitle(f"{object_name} | {title}\n{reference_label}",
                     fontsize=13, fontweight="semibold")
        fig.text(.015, .012, "Independent reconstructed reference; not error-free truth."
                 + (" " + note if note else ""), fontsize=8, color="#4B5964", va="bottom")
        fig.tight_layout(rect=(0, .06, 1, .92), h_pad=1.4, w_pad=1.7)
        for extension in ("png", "svg"):
            path = folder / directory / "figures" / f"{stem}.{extension}"
            kwargs = {"dpi": 170} if extension == "png" else {"metadata": {"Date": None}}
            fig.savefig(path, facecolor="white", **kwargs)
            outputs.append(path.relative_to(folder).as_posix())
        plt.close(fig)

    def phases(ax, labels=False):
        intervals = ((0, train_end / 60, "Earlier training", "#A9C8DF"),
                     (train_end / 60, validation_end / 60, "Validation", "#F2D59D"),
                     (validation_end / 60, test_end / 60, "Held-out test", "#AAD7B7"))
        for left, right, label, color in intervals:
            ax.axvspan(left, right, facecolor=color, alpha=.15, zorder=-10)
            if labels:
                ax.text((left + right) / 2, 1.035, label,
                        transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                        fontsize=8, color="#485762")
        for boundary in (train_end, validation_end):
            ax.axvline(boundary / 60, color="#697A85", linewidth=.8, linestyle=":")
        ax.set_xlim(0, test_end / 60)

    def line(ax, name, x, y):
        ax.plot(x, y, label=MODEL_LABELS[name], color=MODEL_COLORS[name],
                linestyle=MODEL_STYLES[name], linewidth=1.65)

    def metric_bars(ax, field, label, unit, scale=1):
        values = [float(metrics[name]["test"][field]) * scale for name in MODEL_NAMES]
        if not np.isfinite(values).all() or min(values) < 0:
            raise ValueError(f"Invalid nonnegative test metric: {field}")
        bars = ax.bar(np.arange(3), values, width=.64,
                      color=[MODEL_COLORS[name] for name in MODEL_NAMES], alpha=.9)
        ax.bar_label(bars, labels=[f"{value:.5g}" for value in values], padding=4, fontsize=9)
        ax.set(xticks=np.arange(3), xticklabels=["SGP4", "PINN", "Combined"],
               ylabel=unit, title=label, ylim=(0, max(max(values) * 1.25, 1e-12)))
        ax.grid(axis="x", visible=False)

    with plt.rc_context({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.alpha": .18, "grid.linewidth": .6, "legend.frameon": False,
        "figure.facecolor": "white", "savefig.facecolor": "white", "svg.fonttype": "none",
    }):
        for name in MODEL_NAMES:
            directory = name.lower()
            label = MODEL_LABELS[name]
            fig, axes = plt.subplots(3, 2, figsize=(13, 9.8), sharex=True)
            for component, axis_label in enumerate("XYZ"):
                for col, (key, actual, unit, title) in enumerate((
                    ("r", truth_r, "km", "Position in TEME"),
                    ("v", truth_v, "km/s", "Velocity in TEME"),
                )):
                    ax = axes[component, col]
                    ax.plot(minutes, actual[:, component], color="#34483B", linewidth=2.4,
                            alpha=.6, label="ESA reference")
                    line(ax, name, minutes, states[name][key][:, component])
                    phases(ax)
                    ax.set_ylabel(f"{axis_label} ({unit})")
                    if component == 0:
                        ax.set_title(title)
                        ax.legend(fontsize=8, loc="best")
                    if component == 2:
                        ax.set_xlabel("Minutes from shared initial state")
            finish(fig, directory, "01_state_predictions", f"{label}: position and velocity",
                   "Error plots resolve the nearly overlapping tracks.")

            fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.8), sharex=True)
            for ax, (field, scale, ylabel) in zip(axes, (
                ("position_norm_km", 1, "Position error norm (km)"),
                ("velocity_norm_km_s", 1000, "Velocity error norm (m/s)"),
            )):
                line(ax, name, minutes, checked_errors[name][field] * scale)
                phases(ax, labels=ax is axes[0])
                ax.set_ylabel(ylabel)
            axes[0].legend(loc="upper left")
            axes[-1].set_xlabel("Minutes from shared initial state")
            finish(fig, directory, "02_error_norms", f"{label}: errors against ESA",
                   "Shaded splits control hybrid training and selection. Only the final green interval is the primary test.")

            fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
            for component, axis_label in enumerate(("Radial R", "Along-track T", "Normal N")):
                for col, (field, scale, unit, title) in enumerate((
                    ("position_rtn_km", 1, "km", "Position error"),
                    ("velocity_rtn_km_s", 1000, "m/s", "Velocity error"),
                )):
                    ax = axes[component, col]
                    ax.axhline(0, color="#67757C", linewidth=.7)
                    line(ax, name, minutes, checked_errors[name][field][:, component] * scale)
                    phases(ax)
                    ax.set_ylabel(f"{axis_label} ({unit})")
                    if component == 0:
                        ax.set_title(title)
                    if component == 2:
                        ax.set_xlabel("Minutes from shared initial state")
            finish(fig, directory, "03_rtn_errors", f"{label}: signed errors in reference RTN",
                   "Velocity RTN is projected inertial velocity error, not a derivative of rotating coordinates.")

        for name in ("PINN", "COMBINED"):
            rows = histories.get(name, [])
            fig, axes = plt.subplots(1, 2, figsize=(12, 5.8))
            keys = (("training_physics_loss", "validation_physics_loss") if name == "PINN"
                    else ("training_loss", "validation_loss"))
            usable = [row for row in rows if all(
                isinstance(row.get(key), (int, float)) and math.isfinite(row[key])
                for key in ("step", *keys))]
            if usable:
                steps = np.array([row["step"] for row in usable])
                for key, color, label in zip(keys, ("#196DB0", "#C65327"),
                                              ("Training objective", "Validation objective")):
                    loss = np.array([row[key] for row in usable], dtype=float)
                    for ax in axes:
                        ax.plot(steps, loss, color=color, label=label, linewidth=1.5)
                if all(row[key] > 0 for row in usable for key in keys):
                    axes[0].set_yscale("log")
                    axes[0].set_title("All recorded steps (log scale)")
                else:
                    axes[0].set_title("All recorded steps (linear scale)")
                last_half = usable[len(usable) // 2:]
                if len(last_half) > 1:
                    axes[1].set_xlim(last_half[0]["step"], last_half[-1]["step"])
                    tail = [row[key] for row in last_half for key in keys]
                    low, high = min(tail), max(tail)
                    margin = max((high - low) * .12, abs(high) * .05, 1e-15)
                    axes[1].set_ylim(max(0, low - margin), high + margin)
                axes[1].set_title("Later training detail (linear scale)")
                axes[0].legend()
            else:
                for ax in axes:
                    ax.text(.5, .5, "No usable training history was supplied", transform=ax.transAxes,
                            ha="center", va="center")
            for ax in axes:
                ax.set(xlabel="Optimizer step", ylabel="Recorded objective value")
            note = ("Catalog physics objective; ESA observations do not enter standalone PINN training."
                    if name == "PINN" else
                    "Earlier ESA residuals train the correction; validation selects a checkpoint. Test samples never enter this plot.")
            finish(fig, name.lower(), "04_training_history", f"{MODEL_LABELS[name]}: actual training log", note)

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        for ax, (field, scale, ylabel) in zip(axes, (
            ("position_norm_km", 1, "Position error norm (km)"),
            ("velocity_norm_km_s", 1000, "Velocity error norm (m/s)"),
        )):
            for name in MODEL_NAMES:
                line(ax, name, minutes, checked_errors[name][field] * scale)
            phases(ax, labels=ax is axes[0])
            ax.set_ylabel(ylabel)
        axes[0].legend(ncol=3, fontsize=9, loc="best")
        axes[-1].set_xlabel("Minutes from shared initial state")
        finish(fig, "comparison", "01_full_window_errors", "Three-model errors and chronological splits",
               "Hybrid has earlier ESA training information. These curves are not an equal-information comparison.")

        fig, axes = plt.subplots(2, 2, figsize=(12, 8.4))
        for ax, field, title, unit, scale in (
            (axes[0, 0], "position_vector_rmse_km", "Position vector RMSE", "km", 1),
            (axes[0, 1], "position_vector_mae_km", "Position vector MAE", "km", 1),
            (axes[1, 0], "velocity_vector_rmse_km_s", "Velocity vector RMSE", "m/s", 1000),
            (axes[1, 1], "velocity_vector_mae_km_s", "Velocity vector MAE", "m/s", 1000),
        ):
            metric_bars(ax, field, title, unit, scale)
        finish(fig, "comparison", "02_held_out_metric_comparison", f"Held-out test: RMSE and MAE ({test_count} states)",
               "Only samples strictly after the validation boundary are scored. Lower values mean smaller reference error.")

        fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
        for component, axis_label in enumerate(("Radial R", "Along-track T", "Normal N")):
            for col, (field, scale, unit, title) in enumerate((
                ("position_rtn_km", 1, "km", "Position error"),
                ("velocity_rtn_km_s", 1000, "m/s", "Velocity error"),
            )):
                ax = axes[component, col]
                ax.axhline(0, color="#67757C", linewidth=.7)
                for name in MODEL_NAMES:
                    line(ax, name, minutes[test_mask], checked_errors[name][field][test_mask, component] * scale)
                ax.set_ylabel(f"{axis_label} ({unit})")
                if component == 0:
                    ax.set_title(title)
                if component == 2:
                    ax.set_xlabel("Minutes from shared initial state")
        axes[0, 0].legend(fontsize=8, loc="best")
        finish(fig, "comparison", "03_held_out_rtn_errors", "Held-out test: reference RTN errors",
               "All models use the same reference-based RTN frame. Signed model-minus-reference errors.")

        fig, axes = plt.subplots(2, 2, figsize=(12, 8.4), sharex=True)
        for row, (field, scale, quantity, unit) in enumerate((
            ("position_norm_km", 1, "Position", "km"),
            ("velocity_norm_km_s", 1000, "Velocity", "m/s"),
        )):
            counts = np.arange(1, test_count + 1)
            for name in MODEL_NAMES:
                values = checked_errors[name][field][test_mask] * scale
                for col, curve in enumerate((np.sqrt(np.cumsum(values ** 2) / counts),
                                             np.cumsum(values) / counts)):
                    line(axes[row, col], name, minutes[test_mask], curve)
            for col, title in enumerate(("Vector RMSE", "Vector MAE")):
                axes[row, col].set(title=f"{quantity}: {title}", ylabel=f"Error ({unit})")
                if row == 1:
                    axes[row, col].set_xlabel("Minutes from shared initial state")
        axes[0, 0].legend(fontsize=8)
        finish(fig, "comparison", "04_held_out_cumulative_errors", "Cumulative errors within the held-out test",
               "Each point includes only test samples through that time; no earlier training/validation samples.")

    def value(number, scale=1):
        if number is None:
            return "Unavailable"
        try:
            converted = float(number) * scale
        except (TypeError, ValueError):
            return str(number)
        return f"{converted:.8g}" if math.isfinite(converted) else "Unavailable"

    selected_metrics = (
        ("Position vector RMSE (km)", "position_vector_rmse_km", 1),
        ("Position vector MAE (km)", "position_vector_mae_km", 1),
        ("Position 95th percentile error (km)", "position_p95_km", 1),
        ("Maximum position error (km)", "position_max_km", 1),
        ("Final position error (km)", "final_position_error_km", 1),
        ("Velocity vector RMSE (m/s)", "velocity_vector_rmse_km_s", 1000),
        ("Velocity vector MAE (m/s)", "velocity_vector_mae_km_s", 1000),
        ("Maximum velocity error (m/s)", "velocity_max_km_s", 1000),
    )

    def model_table(names, split_name):
        header = "| Metric | " + " | ".join(MODEL_LABELS[name] for name in names) + " |\n"
        header += "| --- | " + " | ".join("---:" for _ in names) + " |\n"
        rows = ["| " + label + " | " + " | ".join(value(metrics[name][split_name].get(key), scale)
                                                       for name in names) + " |"
                for label, key, scale in selected_metrics]
        for component, axis in enumerate(("Radial", "Along-track", "Normal")):
            for statistic in ("rmse", "mae"):
                key = f"position_rtn_{statistic}_km"
                rows.append(f"| {axis} position {statistic.upper()} (km) | " + " | ".join(
                    value(metrics[name][split_name].get(key, [None] * 3)[component]) for name in names) + " |")
        return header + "\n".join(rows)

    details = {
        "SGP4": "Analytical SGP4 propagation uses the saved CelesTrak GP mean elements. It has no neural training step and receives no ESA observations as inputs.",
        "PINN": "A newly trained standalone orbital PINN uses catalog-derived initial states and central-gravity plus J2 differential-equation residuals. Its weights are selected by catalog physics validation. The reference satellite is absent from the catalog splits; ESA future states do not train this model.",
        "COMBINED": "SGP4 plus a newly trained supervised neural residual correction. It predicts position and velocity corrections separately and does not enforce dr/dt = v or orbital dynamics; it is not a physics-constrained combined PINN. The correction uses the earlier ESA training interval, and the next interval selects its checkpoint. Its correction is zero at the common initial time. It receives additional earlier reference information that SGP4 and the standalone PINN do not receive; this difference is part of the experiment.",
    }
    boundaries = (
        f"Training evaluation: 0 < t ≤ {train_end / 60:g} min ({int(train_mask.sum())} scored states); "
        f"validation: {train_end / 60:g} < t ≤ {validation_end / 60:g} min ({int(validation_mask.sum())} states); "
        f"held-out test: {validation_end / 60:g} < t ≤ {test_end / 60:g} min ({test_count} states)."
    )
    limitations = (
        "The independent source is ESA AUX_RESORB, a GNSS restituted orbit estimate, not final AUX_POEORB or error-free truth. "
        "This is a retrospective evaluation: archived GP and ESA products were acquired after the evaluated interval. "
        "Chronological separation prevents held-out reference fitting but does not establish that the exact input products "
        "would have been available in real time. One short satellite arc does not establish debris-wide accuracy or generalization.\n\n"
        "The hybrid has earlier ESA information unavailable to the two standalone baselines. Better or worse test scores therefore "
        "describe these particular configurations, not an equal-information model ranking. Standalone PINN physics includes "
        "central gravity and J2; omitted perturbations and maneuvers can affect errors. The shared GP-derived initial error remains "
        "part of subsequent errors. Frame and reference uncertainty also remain.\n\n"
        "The combined model is a supervised residual corrector. Its position and velocity outputs are predicted separately: "
        "neither dr/dt = v nor orbital dynamics is imposed during its training. A small reference error does not establish "
        "physical consistency, and the combined model must not be interpreted as a physics-constrained PINN. "
        "The runner saves model physics diagnostics in `comparison/physics_diagnostics.json`. Those diagnostics measure "
        "consistency with an approximate central-gravity/J2 model and are not observational accuracy measurements.\n\n"
        "Classification accuracy, precision, recall, F1, ROC/PR AUC, and collision probability are **unavailable** in this "
        "single-object orbit regression experiment. There are no independently labeled conjunction/collision outcomes, "
        "encounter covariances, or object radii. No substitute collision score is presented."
    )
    sources = (
        "CelesTrak GP records use OMM mean-element keywords and the SGP4 convention. "
        "[CelesTrak data-format documentation](https://celestrak.org/NORAD/documentation/gp-data-formats.php).\n\n"
        "Sentinel orbit products are reconstructed from onboard GNSS; RESORB and final precise orbit products have different "
        "processing roles. [Copernicus Sentinel-1 processing documentation](https://sentiwiki.copernicus.eu/web/s1-processing)."
    )
    common_metric_note = (
        "Position arrays and errors are saved in kilometres; velocity arrays and errors are saved in kilometres per second. "
        "Reports/plots convert selected velocity metrics to m/s. Vector RMSE is sqrt(mean(||model − reference||²)); vector MAE "
        "is mean(||model − reference||). Component MAE uses the absolute component. RTN uses the reference orbit's radial, "
        "transverse/along-track and normal directions. Velocity RTN is an inertial velocity error projected onto that frame, "
        "not the derivative of rotating RTN coordinates.\n\n"
        "All state comparisons use identical native reference timestamps in the common TEME frame. The ESA Earth-fixed "
        "position and velocity are both transformed with Earth orientation data; no measured orbit interpolation is required."
    )
    timing_note = str(meta.get("timing_note", "Timing is a diagnostic of this execution; the supplied metadata does not establish a warmed speed benchmark."))

    for name in MODEL_NAMES:
        directory = name.lower()
        own_figures = [path for path in outputs if path.startswith(directory + "/") and path.endswith(".png")]
        figure_lines = "\n\n".join(f"![{Path(path).stem}](figures/{Path(path).name})\n\n"
                                   f"[Scalable SVG](figures/{Path(path).stem}.svg)" for path in own_figures)
        proofs = meta.get("training_proofs", {}).get(name, {})
        proof_lines = []
        if name != "SGP4":
            for label, key in (("Optimizer steps executed", "optimizer_steps_executed"),
                               ("Best checkpoint step", "best_step"),
                               ("Parameter count", "parameter_count"),
                               ("Parameter L2 change", "parameter_l2_change"),
                               ("Training time (seconds)", "training_seconds")):
                if key in proofs:
                    proof_lines.append(f"- {label}: {value(proofs[key])}")
        else:
            proof_lines.append("- Training is not applicable to SGP4.")
        document = f"""# {MODEL_LABELS[name]} results

Satellite: **{object_name}**, NORAD {meta.get('norad_id', 'not supplied')}.
Run window: **{meta.get('start_utc', 'not supplied')} to {meta.get('end_utc', 'not supplied')}**.

{details[name]}

## Primary held-out evaluation

{boundaries}

The primary comparison contains array indices **{test_indices[0]}–{test_indices[-1]}** (zero-based),
exactly **{test_count}** reference states. The preceding boundary sample is excluded from the test aggregate.

{model_table((name,), 'test')}

## Training and runtime

{chr(10).join(proof_lines) or 'Training proof details are recorded in the model folder and run summary.'}

Prediction execution time: {value(meta.get('prediction_timing_seconds', {}).get(name))} seconds.
{timing_note}

## Earlier-interval diagnostics

These earlier intervals are reported separately; their scores are not held-out results for the hybrid.

### Training interval

{model_table((name,), 'train')}

### Validation interval

{model_table((name,), 'validation')}

## Units and definitions

{common_metric_note}

## Graphs

{figure_lines}

## Limits of this result

{limitations}

## Sources and full comparison

{sources}

[Three-model comparison and folder guide](../comparison_report.md).
[Model physics diagnostics](../comparison/physics_diagnostics.json).
"""
        report_path = folder / directory / "report.md"
        report_path.write_text(document, encoding="utf-8")
        outputs.append(report_path.relative_to(folder).as_posix())

    chart_links = "\n\n".join(
        f"![{Path(path).stem}]({path})\n\n[Scalable SVG]({str(Path(path).with_suffix('.svg')).replace(chr(92), '/')})"
        for path in outputs if path.startswith("comparison/") and path.endswith(".png"))
    report = f"""# SGP4, standalone PINN, and combined-model comparison

Satellite: **{object_name}**, NORAD {meta.get('norad_id', 'not supplied')}.
Run window: **{meta.get('start_utc', 'not supplied')} to {meta.get('end_utc', 'not supplied')}**.
Reference: **{reference_label}**. These results are saved locally. After verification, the dashboard can display a published copy under Models.

## Primary result: held-out final interval

{model_table(MODEL_NAMES, 'test')}

Lower errors mean closer agreement with this independent reconstructed reference. No model is assumed to win.
These values summarize the same **{test_count}** test states, zero-based array indices
**{test_indices[0]}–{test_indices[-1]}**; the preceding validation boundary is not included.

## What was run

- **SGP4:** {details['SGP4']}
- **Standalone PINN:** {details['PINN']}
- **Combined:** {details['COMBINED']}

All three predictions begin at the same GP-derived Cartesian position and velocity. The model labels identify
different information budgets: hybrid training sees earlier ESA residuals, while standalone PINN training sees
catalog initial states and physical constraints. The combined model is a trained residual correction, not an
average chosen after examining the held-out test.

## Chronological evaluation

{boundaries}

The shared initial sample at t = 0 is an input-error diagnostic, excluded from aggregate regression scores.
The hybrid's initial correction is constrained to zero; it cannot remove the initial GP error. Training uses
earlier data, checkpoint selection uses only validation, and the final interval is reserved for evaluation.
PINN training/validation curves refer to catalog physics residuals, while hybrid curves refer to its own fitting
objective; the two loss scales should not be compared as if they were the same metric.

## Graphs

{chart_links}

## Training evidence and runtime

| Model | Optimizer steps | Parameter L2 change | Training seconds | Prediction seconds |
| --- | ---: | ---: | ---: | ---: |
"""
    for name in MODEL_NAMES:
        proof = meta.get("training_proofs", {}).get(name, {})
        columns = ["Not applicable", "Not applicable", "0"] if name == "SGP4" else [
            value(proof.get(key)) for key in ("optimizer_steps_executed", "parameter_l2_change", "training_seconds")]
        report += f"| {MODEL_LABELS[name]} | " + " | ".join(columns) + " | " + value(
            meta.get("prediction_timing_seconds", {}).get(name)) + " |\n"
    hardware = meta.get("hardware", {})
    report += f"""
Compute device: **{hardware.get('selected_device', 'not supplied')}**;
accelerator: {hardware.get('accelerator', hardware.get('cpu', 'not supplied'))}.
Training histories, weights and parameter-change evidence are retained with each neural model.
{timing_note}

## Units and error definitions

{common_metric_note}

## Limitations and unavailable evaluations

{limitations}

## Folder guide

- [SGP4 report](sgp4/report.md): SGP4-only predictions, errors, metrics and `figures/`.
- [PINN report](pinn/report.md): standalone neural predictions, model weights, training history, evaluations and `figures/`.
- [Combined report](combined/report.md): SGP4 plus trained neural correction, weights, training history, evaluations and `figures/`.
- `comparison/`: side-by-side metrics and graphs for all three model outputs, plus [physics diagnostics](comparison/physics_diagnostics.json).
- `inputs/` and `reference/`: archived CelesTrak/ESA source inputs and aligned independent reference states.
- The run summary, configuration, hardware report, logs, verification records and source snapshots preserve execution provenance.

Each graph is saved as PNG plus scalable SVG. JSON/CSV/NPZ artifacts created by the runner preserve values
behind the plots. These are actual model/reference results supplied to this report generator; it does not generate
or fit substitute orbit data. No website result pointers are changed by this report generator.

## Sources

{sources}
"""
    report_path = folder / "comparison_report.md"
    report_path.write_text(report, encoding="utf-8")
    outputs.append(report_path.relative_to(folder).as_posix())
    return outputs
