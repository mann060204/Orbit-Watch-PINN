"""Static figures comparing measured model states with an independent orbit reference.

This module plots supplied results only. It does not propagate or generate orbit
data, fit a model, or substitute SGP4 for the independent reconstructed reference.
"""

from pathlib import Path

import numpy as np


def make_truth_figures(folder, seconds, truth_r, truth_v, predictions, errors, metrics, meta):
    """Save publication-readable PNG and SVG figures; return relative file paths.

    State units are TEME kilometres and kilometres/second. RTN errors must already
    be projected onto the independent reference's instantaneous RTN basis. All
    scalar aggregate metrics and cumulative curves exclude the initial sample.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    seconds = np.asarray(seconds, dtype=float)
    if seconds.ndim != 1 or len(seconds) < 2 or not np.all(np.isfinite(seconds)):
        raise ValueError("Plotting requires at least two finite time samples.")
    if seconds[0] != 0 or np.any(np.diff(seconds) <= 0):
        raise ValueError("Time offsets must start at zero and increase strictly.")
    sample_count = len(seconds)

    def array(value, shape, name):
        value = np.asarray(value, dtype=float)
        if value.shape != shape or not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must contain finite values with shape {shape}.")
        return value

    truth_r = array(truth_r, (sample_count, 3), "Reference positions")
    truth_v = array(truth_v, (sample_count, 3), "Reference velocities")
    names = ("SGP4", "PINN")
    colors = {"SGP4": "#D05A25", "PINN": "#176FB0"}
    styles = {"SGP4": "--", "PINN": "-"}
    predicted = {}
    checked_errors = {}
    for name in names:
        predicted[name] = {
            key: array(predictions[name][key], (sample_count, 3), f"{name} {key}")
            for key in ("r", "v")
        }
        checked_errors[name] = {
            key: array(
                errors[name][key],
                (sample_count,) if "norm" in key else (sample_count, 3),
                f"{name} {key}",
            )
            for key in (
                "position_xyz_km", "velocity_xyz_km_s", "position_norm_km",
                "velocity_norm_km_s", "position_rtn_km", "velocity_rtn_km_s",
            )
        }

    folder = Path(folder)
    figure_folder = folder / "figures"
    figure_folder.mkdir(parents=True, exist_ok=True)
    outputs = []
    minutes = seconds / 60.0
    object_name = str(meta.get("object_name", "Evaluation object"))
    reference = str(meta.get("reference_label", "Independent ESA reconstructed orbit"))
    reference_note = "Independent reconstructed reference; not an error-free ground truth."

    def finish(fig, stem, title, note=None):
        fig.suptitle(f"{object_name} | {title}\n{reference}", fontsize=13, fontweight="semibold")
        footnote = reference_note + (f" {note}" if note else "")
        fig.text(0.015, 0.012, footnote, fontsize=8, color="#4B5964", va="bottom")
        # Reserve an explicit footer so long axis labels cannot collide with it.
        fig.tight_layout(rect=(0, 0.048, 1, 0.935), h_pad=1.6, w_pad=1.8)
        for extension in ("png", "svg"):
            target = figure_folder / f"{stem}.{extension}"
            kwargs = {"dpi": 190} if extension == "png" else {"metadata": {"Date": None}}
            fig.savefig(target, facecolor="white", **kwargs)
            outputs.append(target.relative_to(folder).as_posix())
        plt.close(fig)

    def model_lines(ax, field, scale=1.0):
        for name in names:
            ax.plot(minutes, checked_errors[name][field] * scale,
                    color=colors[name], linestyle=styles[name], label=name, linewidth=1.6)

    def grouped_bars(ax, labels, values, ylabel, title):
        x = np.arange(len(labels))
        width = 0.34
        for index, name in enumerate(names):
            bars = ax.bar(x + (index - 0.5) * width, values[name], width,
                          label=name, color=colors[name], alpha=0.90)
            ax.bar_label(bars, labels=[f"{v:.3g}" for v in values[name]],
                         fontsize=8, padding=3)
        ax.set(xticks=x, xticklabels=labels, ylabel=ylabel, title=title)
        ax.margins(y=0.22)
        ax.set_ylim(bottom=0)
        ax.grid(axis="x", visible=False)

    with plt.rc_context({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 11, "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.20, "grid.linewidth": 0.6,
        "legend.frameon": False, "figure.facecolor": "white",
        "savefig.facecolor": "white", "svg.fonttype": "none",
    }):
        # A component view avoids perspective distortion in a 3-D orbit overlay.
        fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
        for component, letter in enumerate("XYZ"):
            for column, (key, actual, unit, quantity) in enumerate((
                ("r", truth_r, "km", "Position"),
                ("v", truth_v, "km/s", "Velocity"),
            )):
                ax = axes[component, column]
                ax.plot(minutes, actual[:, component], color="#263C32", linewidth=2.3,
                        alpha=0.7, label="Independent reference")
                for name in names:
                    ax.plot(minutes, predicted[name][key][:, component], color=colors[name],
                            linestyle=styles[name], linewidth=1.2, label=name)
                ax.set(ylabel=f"{letter} ({unit})")
                if component == 0:
                    ax.set_title(f"{quantity} in TEME")
                    ax.legend(fontsize=8, ncol=3, loc="best")
                if component == 2:
                    ax.set_xlabel("Minutes from shared initial state")
        finish(fig, "01_state_predictions", "Position and velocity predictions",
               "Near-overlapping tracks are resolved in the error figures.")

        fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.5), sharex=True)
        model_lines(axes[0], "position_norm_km")
        model_lines(axes[1], "velocity_norm_km_s", 1000)
        axes[0].set(ylabel="Position error norm (km)", title="Euclidean position error")
        axes[1].set(ylabel="Velocity error norm (m/s)", title="Euclidean velocity error",
                    xlabel="Minutes from shared initial state")
        axes[0].legend(ncol=2)
        finish(fig, "02_error_norms", "Prediction errors against the independent orbit",
               "Initial error is shown; subsequent samples are used for aggregate scores.")

        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        counts = np.arange(1, sample_count)
        for row, (field, scale, quantity, unit) in enumerate((
            ("position_norm_km", 1.0, "Position", "km"),
            ("velocity_norm_km_s", 1000.0, "Velocity", "m/s"),
        )):
            for name in names:
                values = checked_errors[name][field][1:] * scale
                curves = (np.sqrt(np.cumsum(values ** 2) / counts),
                          np.cumsum(values) / counts)
                for col, curve in enumerate(curves):
                    axes[row, col].plot(minutes[1:], curve, label=name,
                                        color=colors[name], linestyle=styles[name], linewidth=1.6)
            for col, measure in enumerate(("Vector RMSE", "Mean error norm (vector MAE)")):
                axes[row, col].set(title=f"{quantity}: {measure}", ylabel=f"Error ({unit})")
                if row == 1:
                    axes[row, col].set_xlabel("Minutes from shared initial state")
        axes[0, 0].legend(ncol=2)
        finish(fig, "03_cumulative_rmse_mae", "Cumulative trajectory error",
               "Each point summarizes all samples after time zero up to that time.")

        for quantity, field, scale, unit, stem in (
            ("Position", "position_rtn_km", 1.0, "km", "04_position_rtn_errors"),
            ("Velocity", "velocity_rtn_km_s", 1000.0, "m/s", "05_velocity_rtn_errors"),
        ):
            fig, axes = plt.subplots(3, 1, figsize=(11.5, 9), sharex=True)
            for component, (ax, label) in enumerate(zip(axes, ("Radial · R", "Along-track · T", "Normal · N"))):
                ax.axhline(0, color="#69757B", linewidth=0.7)
                for name in names:
                    ax.plot(minutes, checked_errors[name][field][:, component] * scale,
                            color=colors[name], linestyle=styles[name], label=name, linewidth=1.6)
                ax.set(ylabel=f"{label} error ({unit})")
            axes[0].legend(ncol=2)
            axes[-1].set_xlabel("Minutes from shared initial state")
            finish(fig, stem, f"{quantity} errors in the reference RTN frame",
                   "Signed model-minus-reference vectors, projected into the reference's instantaneous RTN basis.")

        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
        for ax, (quantity, unit, scale) in zip(axes, (("position", "km", 1.0), ("velocity", "km_s", 1000.0))):
            values = {name: [float(metrics[name][f"{quantity}_vector_{stat}_{unit}"]) * scale
                             for stat in ("rmse", "mae")] for name in names}
            grouped_bars(ax, ["Vector RMSE", "Mean error norm"], values,
                         f"Error ({'km' if quantity == 'position' else 'm/s'})", quantity.capitalize())
        axes[0].legend(ncol=2)
        finish(fig, "06_vector_metric_comparison", "Overall RMSE and MAE comparison",
               "Aggregate scores exclude the initial sample. Mean error norm is the vector MAE convention.")

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for row, (quantity, unit, scale) in enumerate((("position", "km", 1.0), ("velocity", "km_s", 1000.0))):
            for col, statistic in enumerate(("rmse", "mae")):
                values = {name: np.asarray(metrics[name][f"{quantity}_rtn_{statistic}_{unit}"]) * scale
                          for name in names}
                grouped_bars(axes[row, col], ["Radial (R)", "Along-track (T)", "Normal (N)"],
                             values, f"Error ({'km' if quantity == 'position' else 'm/s'})",
                             f"{quantity.capitalize()} component {statistic.upper()}")
        axes[0, 0].legend(ncol=2)
        finish(fig, "07_rtn_metric_comparison", "RTN component RMSE and MAE",
               "Aggregate scores exclude the initial sample; every model uses the same reference RTN frame.")

        # A data-flow figure documents comparison logic without inventing results.
        fig, ax = plt.subplots(figsize=(11, 9.5))
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.axis("off")

        def box(x, y, w, h, text, face="#EDF2F5", edge="#5A6D7B", fontsize=10):
            patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.011,rounding_size=0.012",
                                   linewidth=1.2, edgecolor=edge, facecolor=face)
            ax.add_patch(patch)
            ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)

        def arrow(start, end):
            ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14,
                                        color="#586874", linewidth=1.25))

        box(0.25, 0.90, 0.50, 0.065, "CelesTrak GP / TLE\nOne shared initial Cartesian state", fontsize=10)
        box(0.06, 0.745, 0.30, 0.075, "SGP4 propagation", "#FFF0E6", colors["SGP4"])
        box(0.64, 0.745, 0.30, 0.075, "Frozen, trained PINN\nNo reference fitting", "#E7F2FB", colors["PINN"])
        box(0.06, 0.615, 0.30, 0.065, "Predicted position and velocity\nSame evaluation epochs")
        box(0.64, 0.615, 0.30, 0.065, "Predicted position and velocity\nSame evaluation epochs")
        box(0.275, 0.475, 0.45, 0.070, "Common independent ESA reference\nReconstructed GNSS orbit (RESORB)", "#E7F2EA", "#36754C")
        box(0.25, 0.345, 0.50, 0.075, "Model minus reference\nAligned epochs and common TEME frame")
        for x, label in ((0.04, "Vector RMSE"), (0.38, "Vector MAE"), (0.72, "RTN components")):
            box(x, 0.19, 0.24, 0.065, label)
        box(0.25, 0.045, 0.50, 0.07, "SGP4 versus PINN comparison\nSaved metrics, time series, and graphs")
        arrow((0.38, 0.90), (0.21, 0.82))
        arrow((0.62, 0.90), (0.79, 0.82))
        arrow((0.21, 0.745), (0.21, 0.68))
        arrow((0.79, 0.745), (0.79, 0.68))
        # Both models are evaluated separately against the common reference.
        arrow((0.21, 0.615), (0.285, 0.42))
        arrow((0.79, 0.615), (0.715, 0.42))
        arrow((0.50, 0.475), (0.50, 0.42))
        for x in (0.16, 0.50, 0.84):
            arrow((0.50, 0.345), (x, 0.255))
            arrow((x, 0.19), (0.50, 0.115))
        finish(fig, "08_comparison_workflow", "Independent-reference evaluation workflow",
               "Reference measurements are used only for evaluation; no synthetic ground truth is introduced.")

    return outputs
