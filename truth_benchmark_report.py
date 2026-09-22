"""Readable report for the independent-orbit benchmark, stored only on disk."""
def make_truth_report(folder, summary):
    m = summary["metrics"]
    rows = []
    for label, key, scale in [
        ("Position vector RMSE (km)", "position_vector_rmse_km", 1),
        ("Position vector MAE (km)", "position_vector_mae_km", 1),
        ("Position p95 error (km)", "position_p95_km", 1),
        ("Maximum position error (km)", "position_max_km", 1),
        ("Initial position error (km)", "initial_position_error_km", 1),
        ("Final position error (km)", "final_position_error_km", 1),
        ("Velocity vector RMSE (m/s)", "velocity_vector_rmse_km_s", 1000),
        ("Velocity vector MAE (m/s)", "velocity_vector_mae_km_s", 1000)]:
        rows.append(f"| {label} | {m['SGP4'][key]*scale:.8g} | {m['PINN'][key]*scale:.8g} |")
    for i, axis in enumerate(("Radial", "Transverse", "Normal")):
        for metric in ("rmse", "mae"):
            key = f"position_rtn_{metric}_km"
            rows.append(f"| {axis} position {metric.upper()} (km) | {m['SGP4'][key][i]:.8g} | {m['PINN'][key][i]:.8g} |")
    figures = "\n".join(f"- [{path}]({path})" for path in summary["figures"])
    ratio = m["PINN"]["position_vector_rmse_km"]/m["SGP4"]["position_vector_rmse_km"] if m["SGP4"]["position_vector_rmse_km"] else None
    ratio_text = f"{ratio:.6g}" if ratio is not None else "undefined (SGP4 RMSE is zero)"
    source = summary["source_manifest"]
    report = f"""# SGP4 and PINN versus an independent reference orbit

Satellite: **{summary['object_name']}**, NORAD {summary['norad_id']}.
Window: **{summary['start_utc']} to {summary['end_utc']}**.
Samples: {summary['sample_count']} states at {summary['step_seconds']:g}-second intervals;
aggregate metrics use {summary['aggregate_samples_per_model']} future states, with the initial error reported separately.

## Your requested comparison flow

```text
                   Same initial CelesTrak GP / TLE-equivalent elements
                                  /                   \\
                               SGP4                Frozen PINN
                                |                       |
                         predicted r,v            predicted r,v
                                |                       |
                       subtract ESA reference at identical UTC times
                       (both r and v transformed into common TEME frame)
                                  \\                   /
                                RMSE, MAE, RTN errors
                                         |
                               Side-by-side comparison
```

The independent reference does not train, initialize, or correct either prediction.
It is used only after their trajectories are computed. Both start from exactly the same
SGP4-derived Cartesian initial state from the common GP record.

## Measured errors against ESA

| Metric | SGP4 | PINN |
| --- | ---: | ---: |
"""+"\n".join(rows)+f"""

The PINN/SGP4 position RMSE ratio is {ratio_text}; ratios above 1 mean the PINN has
larger position error in this benchmark. This is one external satellite and one
window, not a general ranking for the debris catalog.

All values come from actual propagator and saved neural-model executions compared
with independently reconstructed ESA states. No fabricated observations or scores
were used. Classification accuracy, precision, recall, F1, and collision probability
are not defined by this single-object regression benchmark; their unavailable status
and reasons are in `evaluation/unavailable_classification_metrics.json`.

## Reference quality and scope

The source is **ESA AUX_RESORB**, a GNSS-based NRT restituted orbit estimate, with
NOMINAL quality flags. It is independent of the project's SGP4/PINN outputs. It is
**not final AUX_POEORB and not error-free physical truth**. The output folder uses
the requested ground-truth terminology, but every reported error is relative to
this qualified reconstructed reference. A service accuracy specification is not a
measured uncertainty or covariance for this particular file.

This is retrospective: the GP elements and ESA file were fetched after the evaluated
window. The GP epoch ({summary['gp_epoch_utc']}) precedes all evaluation samples, but
availability of this exact GP set at that historical instant has not been established.
This run cannot demonstrate real-time forecast skill or debris collision detection.

The frozen model comes from the earlier saved comparison run. Sentinel-1D is absent
from all of its original object splits. Its checkpoint was fixed before this reference
was acquired. There is no ESA-based training, tuning, model selection, or bias correction.
Original training proof, history, splits, summary, and weights are copied into `trained_model/`.
PINN physics remains central gravity plus J2; drag, higher harmonics, third bodies,
and maneuvers are omitted. The known reference and model limitations remain relevant.

## Time, units, and frame handling

- The initial GP input is unmodified CelesTrak JSON with OMM mean-element conventions.
  A derived TLE encoding is included for inspection; it is not separately downloaded
  and is not used in place of the original input, avoiding an extra rounding change.
- Each evaluation timestamp is an actual ESA OSV timestamp: no orbit interpolation,
  no extrapolation, and no alignment by nearest unmatched time.
- ESA EARTH_FIXED positions/velocities in metres and metres per second are converted
  to kilometres and kilometres per second and transformed from ITRS to TEME with Astropy.
  Velocity transformation includes the rotation of the Earth-fixed frame.
- UT1 minus UTC comes from the corresponding ESA OSV. Polar motion comes from the
  saved IERS finals2000A file. Polar-motion status counts are
  `{summary['frame_transformation']['polar_motion_status_counts']}`; flag 0 means IERS B,
  1 means IERS A, and 2 means IERS A prediction. Reference and Earth-orientation
  uncertainties are not set to zero.
- Frame round-trip checks establish numerical consistency, not absolute frame accuracy.
  Terrestrial-realization and Earth-orientation residuals are not separately calibrated.
- The RTN basis is built from the independent reference: R = r / |r|,
  N = (r × v) / |r × v|, T = N × R. Both models use that same basis at each timestamp.
  Velocity RTN errors are projections of inertial velocity differences; they are not
  derivatives of coordinates in the rotating RTN basis.

## Metric definitions

For future samples i = 1..N, position error is e_i = r_model,i − r_reference,i.
Vector RMSE = sqrt(mean(|e_i|²)); vector MAE = mean(|e_i|).
RTN component RMSE = sqrt(mean(e_axis,i²)); component MAE = mean(abs(e_axis,i)).
Velocity errors use the same definitions with v. Initial input error is included
in subsequent prediction errors; no subtraction or initial-state correction hides it.
Cumulative curves use future samples up to the plotted timestamp and omit time zero.

## All files in this folder

- `inputs/`: exact CelesTrak response/selected record, ESA ZIP and EOF, source manifest,
  derived TLE, shared initial state, and the Earth-orientation file.
- `reference/`: independent orbit in ITRS/TEME, source sample indices, frame checks,
  and Earth-orientation values at every evaluation time.
- `predictions/`: full SGP4/PINN r,v arrays and CSV containing both predictions plus ESA.
- `evaluation/`: all metrics in JSON/CSV, the side-by-side table, per-time signed
  Cartesian/RTN errors, error arrays, and verification checks.
- `trained_model/`: actual frozen weights and original training provenance.
- `source_code/`: code/dependency snapshots. `summary.json`, `hardware.json`,
  `run.log`, `status.json`, and `file_checksums.json` preserve run provenance.

## Graphs

Each plot is saved as PNG and scalable SVG, including your workflow diagram.

{figures}

## Reproduce this benchmark

From the main project:

```powershell
.\\run_truth_benchmark.ps1 --inputs "{(folder/'inputs').as_posix()}" --model-run "{(folder.parent.parent/'comparison_results'/summary['model_training_run']).as_posix()}" --hours {summary['horizon_hours']:g} --step-seconds {summary['step_seconds']:g} --eop-file "{(folder/'inputs/finals2000A.all').as_posix()}"
```

The original model run supplies its unchanged checkpoint and training provenance.
The saved inputs/EOP file avoid changing reference data during reproduction.
No website files or dashboard result pointers are modified.

## Sources

- [CelesTrak GP query]({source['gp_source']['url']})
- [ESA orbit product]({source['reference_source']['url']})
- [Copernicus orbit-product definitions](https://sentiwiki.copernicus.eu/web/s1-processing)
- [Astropy satellite frame handling](https://docs.astropy.org/en/stable/coordinates/satellites.html)
- [IERS Earth-orientation data](https://datacenter.iers.org/data/9/finals2000A.all)
"""
    (folder/"comparison_report.md").write_text(report, encoding="utf-8")
