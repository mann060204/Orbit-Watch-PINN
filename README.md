# PINN space debris collision prediction

Step 1 collects the latest available orbital elements from CelesTrak. Step 2 uses
SGP4 to propagate them, screen close approaches, save reproducible results, and show
a live local dashboard. Step 3 trains a physics-informed neural network on actual
catalog initial conditions, predicts close approaches, and saves measured evaluations.

Double-click **StartDashboard.bat** to start the local dashboard and open it in
your default browser. Keep its console window open while using the dashboard;
press Ctrl+C to stop it. If the dashboard already runs on port 8787, the launcher
opens that instance. Keep the BAT file in this project folder, or create a Windows
shortcut to it.

## Separate SGP4, PINN, combined and comparison outputs

The dashboard opens with **Orbital view**, followed by **Model training** and
**Orbital evaluation**. Use the numbered navigation or sidebar to jump between
them. Object history and Gemini chat expand beneath the globe.

In **Model training**, choose a completed experiment to inspect both networks'
training histories and download their weights. The same experiment supplies the
**Comparison**, **SGP4**, **PINN** and **Combined** evaluation tabs below. Choose
the test, validation or training interval to view actual errors, RMSE/MAE/RTN
metrics, saved trajectories, graphs, hardware and verification results. Individual
artifacts and the complete run ZIP are downloadable. Live SGP4 screening and
close-approach records remain available on the same page.

Open **http://127.0.0.1:8787/#models** and refresh the page to load the new tabs.
These are dated benchmark results, separate from the live CelesTrak catalog.

Double-click **RunThreeModels.bat** to train both neural networks and reproduce
the three-model experiment on the saved real CelesTrak and ESA inputs. After
saved weights and metrics pass replay checks, results appear on the dashboard. This
retrospective benchmark requires the existing files in
`reference_inputs/sentinel_benchmark/` and the verified September 11 catalog in
`comparison_results/20260911T100012_038157Z/`. It does not download new data or
present the archived September 13 orbit as a live observation.

Completed run: `model_results/20260914T093047_375528Z/`.

```text
model_results/<UTC run ID>/
  comparison_report.md     # Start here: methods, held-out results and graph links
  sgp4/                    # SGP4-only states, errors, metrics and figures
  pinn/                    # Fresh PINN weights, training logs, states and figures
  combined/                # SGP4 + trained neural correction, weights and figures
  comparison/              # Shared tables, comparison figures and replay checks
  inputs/                  # Exact CelesTrak/ESA inputs and earlier training catalog
  reference/               # Independent ESA orbit in aligned TEME coordinates
  source_code/             # Code and dependency snapshots
  hardware.json            # Detected device and training configuration
  config.json, summary.json, file_checksums.json
```

This computer ran 6,000 standalone PINN optimizer steps in 167 seconds and 3,000
neural-correction steps in 7.9 seconds on the CPU. Both networks were trained
from scratch. SGP4 itself is a deterministic propagator and needs no neural
training. The run saves 15 graphs in both PNG and SVG, four reports, CSV/JSON
metrics, NPZ state/error arrays, weights and actual training histories.

All models start from the same Sentinel-1D CelesTrak GP-derived position and
velocity. Standalone PINN training uses earlier catalog initial states and
central-gravity/J2 physics, with Sentinel-1D excluded. The combined model is
**SGP4 plus a supervised neural residual correction**: it uses minutes 1–90 of
the ESA arc for training and minutes 91–120 for checkpoint selection. It
predicts position and velocity corrections separately without imposing orbital
dynamics or `dr/dt = v`. It is not a physics-constrained combined PINN.

The primary comparison uses exactly the same 60 reserved states at minutes
121–180 for all models. The combined model has earlier ESA information that
the standalone baselines do not receive, so this is a comparison of those
configurations, not an equal-information ranking of algorithms.

| Held-out metric | SGP4 | PINN | SGP4 + neural correction |
| --- | ---: | ---: | ---: |
| Position RMSE (m) | 503.486 | 3715.756 | 275.348 |
| Position MAE (m) | 436.781 | 3698.599 | 237.148 |
| Velocity RMSE (m/s) | 0.472051 | 4.187752 | 0.268482 |

The independent reference is ESA GNSS **AUX_RESORB**, a restituted orbit
estimate, not final precise POEORB or error-free truth. This single-satellite,
three-hour retrospective experiment does not validate debris collision
predictions. Accuracy, precision, recall, F1 and collision probability are
explicitly unavailable without independently labeled encounters, covariances
and physical radii. RMSE, MAE, component/RTN errors and measured runtimes are
computed from actual model/reference outputs.

Replay both saved neural networks, rederive SGP4, and verify every split metric
and artifact checksum without retraining:

```powershell
.\.venv\Scripts\python.exe verify_three_model_run.py "model_results\20260914T093047_375528Z"
```

The launcher accepts `--pinn-steps`, `--correction-steps`, `--seed`, `--input-dir`
and `--training-run`. Changing the configuration creates a new result folder.
`model_results/latest.json` points to the last completed and replay-verified run.

To publish existing completed runs without training again:

```powershell
.\.venv\Scripts\python.exe publish_model_results.py
```

The publisher verifies run checksums and replay status, then copies only scientific
artifacts into `dashboard/model-results/` and writes a run index. The running
dashboard serves those static assets, so this results view needs no server
restart. Original experiment folders stay unchanged; older reports retain their
creation-time statements that they were initially saved only to disk. The
publication record documents their later display on the dashboard. API keys
and project files outside the scientific run are not copied.

## Object history and Gemini chat

Select a point on the globe or search a NORAD ID. Below the object selector,
**History & orbital context** shows sourced mission/breakup information, current
SGP4 orbital facts, locally saved element updates, and that object's latest saved
screening events. Iridium 33, Cosmos 2251, Fengyun-1C, and the ISS have bundled
NASA/ESA histories. Other objects show their available catalog data and identify
missing biographical information. A breakup family's history is identified as
family context, rather than an independently verified biography of each fragment.

To enable **Ask Orbital Watch**:

1. Start the dashboard using `StartDashboard.bat`. After a code update, stop an
   older dashboard console with Ctrl+C and start the BAT again.
2. Click **Connect Gemini** (or **Gemini settings**), enter your Gemini API key,
   and save. The default model is `gemini-3.1-flash-lite`.
   Use a model available to your API project.
   Create a Gemini key in [Google AI Studio](https://aistudio.google.com/apikey).
   Saving settings does not verify API access; that is checked on your first question.
   If Gemini says the server needs restarting, stop the running dashboard server,
   run `StartDashboard.bat` again, and refresh the page.
3. Ask a question. Each request sends the selected object's current context,
   your question, and a short conversation history to Google.
   Gemini API billing and quotas apply. Object details remain available without a key.

The backend uses Google's documented
[Gemini chat-completions endpoint](https://ai.google.dev/gemini-api/docs/openai).
Requests use bounded output and no automatic retries. Google's API data policies apply; its
[free-tier pricing](https://ai.google.dev/gemini-api/docs/pricing) lists content
as used to improve its products. This chat has no live web-search tool; answers
about the selected object use the supplied data and linked sources. Missing
collision covariances, physical dimensions, or observational accuracy cannot be
replaced by an LLM estimate. Provider errors are displayed without fabricated
AI replies. Cloud generation runs on Google's servers; this computer hosts the
dashboard and its orbital calculations.

If chat reports a **rate or quota limit**, check your model's usage and limits in
Google AI Studio. Temporary limits show the minimum retry delay when available;
daily or free-tier quota may require waiting for a reset or enabling billing.
See the [Gemini limits guide](https://ai.google.dev/gemini-api/docs/rate-limits).

Keys entered in the form are encrypted with Windows DPAPI for the current
Windows user and stored in `.local/chat_gemini_config.json`. Existing Gemini
settings use the same file. This folder is git-ignored
and is not served by the dashboard or included in result downloads. The server
never returns saved keys to the browser. **Remove saved Gemini key** deletes this
saved connection. Alternatively, start the server with `GEMINI_API_KEY` and
optional `GEMINI_MODEL` environment variables. Other provider settings are not read.
Saved settings take precedence; removing a saved key does not unset an environment key.
Conversation messages are kept in browser memory for the current page session.

History sources are in `object_knowledge.json`; context extraction is in
`object_context.py`; API and credential handling are in `chat_service.py`.
This feature reads the dashboard's catalog and saved SGP4 screening. Verified
model comparison results are available in the dashboard's Models section.

## Compare both models against an independent orbit

This follows the requested flow: one initial CelesTrak GP/TLE-equivalent input,
SGP4 and PINN position/velocity predictions, then separate errors against an
independent ESA orbit at identical timestamps. Results stay in a single local
folder; the website is not updated.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-truth.txt
.\run_truth_benchmark.ps1
```

Install the base and PINN dependencies first on a new machine. The command uses
the saved Sentinel-1D inputs in `reference_inputs/sentinel_benchmark/`, the frozen
model from `comparison_results/latest.json`, and the saved IERS Earth-orientation
file when present. These are dated benchmark inputs, not a fresh live download.
`--inputs`, `--model-run`, `--hours`, `--step-seconds`, and `--eop-file` select
explicit reproducible inputs. The completed report records the exact command.

The independent reference is **ESA AUX_RESORB**, reconstructed from onboard GNSS,
with NOMINAL quality flags. It is a restituted estimate, **not the final precise
AUX_POEORB product or error-free physical truth**. The September 13, 2026 test
uses Sentinel-1D (NORAD 66315), which was absent from the earlier model's original
data splits. The model remains frozen: ESA observations are used only to evaluate
the predictions. This is one retrospective one-hour test, not an operational
debris-collision validation.

Each completed run saves everything under `ground_truth_results/<UTC run ID>/`:

- `comparison_report.md`: comparison, metric definitions, limitations, and sources.
- `figures/`: eight graphs, each in PNG and SVG, covering position/velocity,
  error curves, cumulative RMSE/MAE, RTN errors, metric comparisons, and the workflow.
- `evaluation/`: metrics in CSV/JSON, signed Cartesian/RTN errors at every time,
  error arrays, verification checks, and unavailable classification metrics.
- `predictions/` and `reference/`: both computed trajectories and the independent
  ESA orbit in matched units/frames, with the Earth-orientation transformation audit.
- `inputs/`, `trained_model/`, and `source_code/`: the real source data, hashes,
  frozen weights, training history, dependency versions, and reproducible code.

`ground_truth_results/latest.json` identifies the latest completed benchmark.
Position/velocity RMSE, MAE, p95, maximum, signed bias, and RTN components are real
measurements against the independent reference. Accuracy, precision, recall, F1,
and collision probability remain unavailable because a single orbit supplies
neither labeled conjunction events nor collision covariances and physical radii.

Completed independent benchmark: `ground_truth_results/20260913T142535_748588Z/`.
It evaluates 60 future states from 00:11:14.932121 to 01:10:14.932121 UTC on
September 13, 2026, plus a separately reported initial state one minute earlier.

| Metric against ESA RESORB | SGP4 | PINN |
| --- | ---: | ---: |
| Position vector RMSE (m) | 434.014 | 394.841 |
| Position vector MAE (m) | 426.296 | 377.794 |
| Velocity vector RMSE (m/s) | 0.443633 | 0.385813 |
| Final position error (m) | 348.453 | 515.150 |

PINN has lower aggregate error in this window but higher final position error.
The result does not establish a general ranking across objects and horizons.

To independently recompute metrics, replay both saved models, and verify the
original input/source/output checksums without downloading data or training:

```powershell
.\.venv\Scripts\python.exe verify_truth_run.py "ground_truth_results\20260913T142535_748588Z"
```

The audit adds `evaluation/artifact_verification.json` and its verifier source
copy to the result folder after all original checks pass.

## Compare SGP4 and PINN without the website

Run the dedicated comparison from PowerShell:

```powershell
.\run_comparison.ps1
```

This trains a fresh PINN and runs both models using one CelesTrak snapshot, one
eligible object set, identical initial states, and a common forecast window. It
uses the existing two-hour download cache. Defaults are 6,000 optimizer steps,
a one-hour horizon, 60-second neural steps, and a 5 km threshold. Hardware is
checked before training. The current CPU build works on this computer.

Every output is saved under `comparison_results/<UTC run ID>/`. Only
`comparison_results/latest.json` is updated. The comparison does not write the
dashboard's data/results pointers, add routes, or publish results on the website.
The CelesTrak source cache is shared to respect the download policy.

```text
comparison_results/<UTC run ID>/
  comparison_report.md              # Readable side-by-side results and interpretation
  data/snapshots/<snapshot>/         # Exact downloaded GP data and original timestamps
  sgp4/                             # SGP4 trajectories, encounters, error codes and audits
  pinn/                             # Trained weights, trajectories, encounters, training logs
  evaluation/
    model_comparison.csv            # Timings, state counts and event counts for both models
    all_metrics.csv, metrics.json   # Complete evaluation values and unavailable fields
    per_object_errors.csv           # Position/velocity error and train/validation/test membership
    errors_by_forecast_time.csv      # Error growth across the same forecast window
    trajectory_differences.npz       # Position/velocity vector differences and signed RTN errors
    pair_agreement.csv              # Positive-pair agreement and disagreements
    pair_universe.json              # Complete eligible pair universe, including negatives
    event_matches.csv               # One-to-one matched and unmatched encounter episodes
    propagation_benchmarks.csv      # Repeated measured timings, alternating model order
    artifact_verification.json      # Reloaded model, same inputs, and website isolation checks
  figures/                          # Static comparison and error charts
  source_code/                      # Source and dependency snapshots
  initial_conditions.npz            # Shared Cartesian initial states and catalog IDs
  data_splits.json                  # Grouped object partitions
  eligible_objects.json            # Common modeled catalog records
  excluded_objects.json            # Excluded catalog records and reasons
  shared_orbit_pairs.json           # Pairs whose independent separation is unresolved
  config.json, hardware.json        # Settings and detected hardware
  summary.json, file_checksums.json # Result summary, provenance and artifact hashes
  run.log, status.json              # Progress and completion state
```

The report distinguishes three kinds of evaluation:

- **SGP4/PINN agreement:** trajectory RMSE, MAE, p95/max errors, TEME and
  radial/transverse/normal components, accuracy, precision, recall, F1, specificity,
  balanced accuracy, NPV, FPR/FNR, MCC, and confusion counts. All-object and test-only
  results are separate. SGP4 is the comparison reference, not measured ground truth.
- **Measured execution and physics diagnostics:** training time, repeated propagation
  times, screening time, finite states, central+J2 residuals, and invariant drift.
  SGP4 includes different perturbations, so central+J2 consistency does not establish
  which model is more accurate observationally. Screening runtimes include each
  routine's different diagnostic work; propagation timings exclude disk I/O and training.
- **Unavailable observational evaluations:** neither model receives an invented
  real-world accuracy score. ROC/PR AUC, Brier score, and collision probability are
  null without the required labels, calibrated scores, covariance, and physical radii.

The encounter comparison uses one-to-one same-pair matching within two time steps,
retains unmatched events from both models, and distinguishes episode agreement
from pair flags across the whole window. High accuracy can be dominated by the
large number of negative pairs. A test split without positive reference pairs has
undefined precision/recall, not demonstrated perfect detection.

For reproduction, select the exact saved snapshot and start time shown in the
report. `--snapshot` bypasses new downloads and copies that input into the new run:

```powershell
.\run_comparison.ps1 --snapshot "path\to\saved\snapshot" --start "2026-09-11T10:00:00Z"
```

Other options: `--steps`, `--hours`, `--segment-seconds`, `--threshold-km`, `--seed`,
and `--benchmark-repeats`. Use one comparison process at a time. Failed runs retain
their error log and do not replace the latest completed comparison.

## Run the PINN (step 3)

The first real training run is complete in
`pinn_results/20260911T091655_561251Z/`. Its reports, training evidence,
error plots, encounter tables and model files remain available on disk.
The dashboard's Models section displays PINN results from verified three-model
experiments. This earlier standalone catalog run remains available in its folder.

This computer has an Intel i5-1135G7, Iris Xe integrated graphics, and approximately
19.7 GiB usable RAM. CUDA is unavailable. The run used PyTorch CPU, two threads,
float64, batches of 256, and a network with 5,334 parameters. It executed 6,000
optimizer steps in 105 seconds, using about 321 MB of peak sampled process memory.
Hardware is inspected again before each run, and the selection is saved.

The actual run modeled 1,997 fresh, valid objects from the 2,687-record CelesTrak
download on 2026-09-11. It excluded 690 objects, primarily for element age over
72 hours, and separated 31 pairs with identical orbital elements. Its forecast
window is 09:16:55–10:16:55 UTC on that date; saved forecasts are not silently
presented as current observations.

| Measured result | First completed run |
| --- | --- |
| Neural close approaches within 5 km | 17 |
| SGP4 reference close approaches | 16 |
| Closest neural approach | 1.10876 km |
| All-pair precision / recall vs SGP4 | 88.24% / 93.75% |
| All-pair F1 vs SGP4 | 0.9091 |
| True positives / false positives / false negatives | 15 / 2 / 1 |
| Held-out test position vector RMSE vs SGP4 | 3.13929 km |
| Held-out final-time position vector RMSE | 6.96293 km |
| Held-out maximum position error | 60.71074 km |
| Observed collision accuracy and collision probability | Unavailable |

The classification scores above cover all eligible pairs, including objects used
for training. Separate test-only pair metrics are saved; that test-only set has
zero positive reference encounters, so its precision and recall are undefined.
All-pair accuracy is 99.999849%, dominated by almost two million negative pairs;
it must not be read as evidence of reliable collision prediction. The measured
position errors are substantial relative to the 5 km screening threshold.

For a fresh standalone catalog screening run, use PowerShell:

```powershell
.\run_fetch.ps1
.\run_pinn.ps1
.\.venv\Scripts\python.exe verify_pinn_run.py
```

The first command refreshes data subject to the existing two-hour cache. The
training command uses the latest downloaded snapshot. Defaults are 6,000 steps,
one hour, 60-second neural segments, and a 5 km screening threshold. CLI settings
include `--steps`, `--hours`, `--segment-seconds`, `--threshold-km`, `--seed`,
`--start`, and `--covariances`. Longer horizons require additional validation.
Do not run overlapping CLI jobs. Neural CLI jobs are independent of the dashboard.
Automatic CelesTrak refresh does not automatically retrain the network.

Install the neural dependency after the base dependencies on a new machine:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-pinn.txt
```

The supplied PINN dependency pins the CPU build tested on this computer. A CUDA
machine requires a compatible CUDA PyTorch build to use that accelerator.

### How the neural predictions are produced

1. Convert downloaded GP elements into initial TEME position/velocity states with
   SGP4 at the selected start time. These are orbit estimates, not measured tracks.
2. Split objects by identical-orbit groups into training, validation, and test
   partitions (1,400 / 298 / 299 objects in the first run).
3. Train a three-layer tanh neural network through automatic differentiation of
   the central-gravity and J2 equations. A Taylor trial solution enforces exact
   initial conditions; the network learns higher-order position/velocity corrections.
4. Select the checkpoint by validation physics residual, then roll its learned
   short-time flow map forward in 60-second intervals. No future SGP4 states or
   measured future tracks are used for training or checkpoint selection.
5. Screen the neural trajectories and refine candidate encounter times with the
   neural model. Generate future SGP4 tracks separately after training, then
   calculate reference agreement, regression errors, and classification metrics.

These are outputs from an actual optimization run, not prefilled or mocked
dashboard values. They remain model predictions. Physics collocation points are
equation constraints, not observed trajectory samples. The model omits drag,
J3/J4, third-body forces, and maneuvers; TEME is treated as quasi-inertial over this
short window. This implementation does not establish operational collision accuracy.

### All neural results in one folder

`pinn_results/latest.json` identifies the last complete neural run, separately
from the SGP4 results. Each `pinn_results/<UTC run ID>/` contains:

```text
model.pt                              # Actual trained weights and checkpoint metadata
input_snapshot/, initial_conditions.npz # Exact catalog and starting states
data_splits.json                       # Disjoint object partitions
hardware.json, config.json             # Device selection and training settings
training_history.csv, .jsonl           # Measured losses, timing, sampled memory
training_proof.json                    # Steps, parameter changes, best validation loss
pinn_trajectories.npz                  # Neural positions, velocities, IDs and times
pinn_conjunctions.csv, .json            # Neural encounter times, distances and speeds
sgp4_reference_trajectories.npz         # Evaluation reference only
sgp4_reference_events.json             # Separate SGP4 encounter reference
metrics.json, metrics.csv              # All metrics and unavailable values/reasons
per_object_evaluation.csv              # Per-object regression errors and split
reference_errors.npz                   # Position/velocity error at each sample
pair_classification_vs_sgp4.csv         # Flagged pair comparisons
pair_universe.json                     # Complete eligible pair universe definition
event_comparison_vs_sgp4.csv            # Matched encounter time/distance differences
pinn_candidate_audit.csv               # Refined intervals, including rejected candidates
excluded_objects.json                  # Excluded objects and reasons
shared_orbit_pairs.csv                 # Indistinguishable modeled orbits
figures/                               # Actual loss, error, confusion and event plots
report.md, report.html                 # Readable experiment report
summary.json, status.json, run.log      # Completion, provenance and code hashes
artifact_verification.json             # Written by verify_pinn_run.py after replay
```

ROC AUC, PR AUC, and Brier scores are unavailable because there are no calibrated
probability scores. Observational collision metrics remain null without independent
outcomes. SGP4-reference precision/recall measure model agreement, not observational
accuracy. Unit tests use synthetic fixtures only to check mathematics and code;
those fixtures are never included in the saved experiment results.

The checkpoint replay utility reloads the weights with `weights_only=True`,
reproduces all saved neural states, recomputes test RMSE and confusion counts,
and checks the recorded source hashes. It does not retrain the model.

### Collision probability with real uncertainty inputs

No covariances or physical object radii are present in the downloaded GP data.
Consequently every current neural event has `collision_probability: null`.
`collision_probability.py` implements an optional 2D encounter-plane Gaussian
integral. It requires encounter-time position covariance matrices for both objects,
a justified combined hard-body radius, matching IDs/time, and input provenance.
It never substitutes neural residuals for physical orbital covariance.

Use `--covariances path/to/real_covariances.json` only with actual supplied values.
The JSON object contains an `events` array. Each entry must contain:

| Field | Required meaning |
| --- | --- |
| `object1_id`, `object2_id`, `tca_utc` | Same object order and encounter, TCA within 1 second |
| `frame` | `TEME` |
| `units` | `km,km^2` |
| `covariance1_km2`, `covariance2_km2` | Symmetric positive-semidefinite 3×3 matrices at TCA |
| `combined_hard_body_radius_km` | Positive sum of justified physical radii |
| `cross_covariance_assumption` | `independent` (only supported correlation model) |
| `source`, `radius_source` | Nonempty covariance and object-size provenance |

The optional method assumes a brief, approximately rectilinear encounter and
independent Gaussian errors; relative speeds below 0.01 km/s are rejected. It
does not automatically account for PINN model error. Probabilities would remain
conditional on these assumptions and the supplied inputs, not automatically
calibrated collision forecasts.

References: [PINN formulation and automatic differentiation](https://arxiv.org/abs/1711.10561),
[NASA close-approach risk assessment](https://www.nasa.gov/cara/step-2-close-approach-risk-assessment/),
[NASA conjunction assessment handbook](https://ntrs.nasa.gov/api/citations/20230002470/downloads/CA_Handbook_CM%20Version%202-24-23.docx.pdf).

## Start the dashboard (step 2)

The local environment is installed and the first full 24-hour analysis is saved.
From this project folder in PowerShell:

```powershell
.\run_dashboard.ps1
```

Open [Orbital Watch](http://127.0.0.1:8787). Keep the process running while using
the dashboard; press Ctrl+C in its terminal to stop it. It binds to this computer's
loopback interface. A different port can be selected with `--port 8788`.

The dashboard provides current SGP4 positions every five seconds, an interactive
TEME orbital view, object search and orbit inspection, close-approach filtering,
encounter distance curves, analysis controls, evaluation status, and result downloads.
The globe is an inertial coordinate view, not a geographic ground-track map.
Displayed altitude is geocentric radius minus 6378.137 km, not geodetic altitude.

While the server runs, automatic refresh checks the original download times and
fetches due groups after two hours, then screens the new snapshot. The refresh
toggle can disable this behavior. Start with `--no-auto-refresh` to disable it on
launch. Any refresh failure disables automatic refresh and exposes the error;
investigate before enabling it again. The last complete result remains available.
Manual refresh also honors the same cache. Data is never fetched every five seconds.

For a new installation, use Python 3.12 or newer for the pinned step-2 environment
(this project was verified with Python 3.12). Install the dependencies first:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
```

Run without the dashboard:

```powershell
.\run_analysis.ps1
.\run_analysis.ps1 --hours 6 --step 30 --threshold-km 5
.\run_analysis.ps1 --include-stale
```

Defaults are a 24-hour forecast from now, 60-second samples, a 5 km close-approach
threshold, and a 72-hour element age limit at analysis start. All initialized objects
are propagated and saved. Stale elements and any object with a propagation error
are excluded from screening unless explicitly allowed (stale elements only).
Identical modeled orbit pairs are saved separately in `shared_orbit_pairs.csv`;
their independent separation cannot be resolved from these elements. They are not
counted as collision candidates. Other co-orbiting entries can still appear and
need interpretation. Run only one analysis process at a time.

The script uses WGS72, direct OMM initialization, and split Julian dates. A swept
chord spatial search considers motion between samples; per-object padding is
`A * dt^2 / 8`, with an assumed acceleration bound `A = 0.05 km/s²`. Candidate
intervals are refined by direct SGP4 distance minimization, including interval
endpoints. This is research screening, not a proof of exhaustive conjunction
detection. Reducing the time step is useful for convergence studies. A 1 ms
minimizer tolerance describes numerical refinement, not forecast accuracy.

The sgp4 library's internal catalog-number encoding is still limited. For IDs
above 339999, this project uses a neutral internal propagator ID while retaining
the full real ID in every catalog row, state array, event, and report. Catalog IDs
are metadata, not inputs to the orbit equations.

## All analysis results in one folder

`results/latest.json` identifies the last complete run. Each run has its own folder:

```text
results/<UTC run ID>/
  input_snapshot/                    # Exact source data used in this run
  config.json, summary.json           # Settings, time window, versions, code hashes
  trajectories.npz                   # Every propagated position, velocity, time, ID, error
  objects_and_initial_states.csv      # Catalog, eligibility, and initial TEME states
  conjunctions.csv, conjunctions.json # Refined times, miss distances, relative speeds
  candidate_audit.csv                 # Candidate episodes including rejected candidates
  shared_orbit_pairs.csv              # Identical modeled orbits, separated from alerts
  excluded_objects.csv, .json         # Age, initialization, or propagation exclusions
  propagation_errors.csv             # Invalid state counts and SGP4 error codes
  refinement_errors.json              # Any failures during closest-approach refinement
  metrics.json, metrics.csv           # Measured metrics and explicit unavailable values
  validation.json                    # Published Vallado numerical reference check
  labels_template.json               # Schema for independent evaluation labels
  figures/screening_summary.png       # Freshness, distance, and timeline charts
  report.md, run.log, status.json     # Readable report, process log, completion status
```

The complete run is available as a ZIP from the dashboard. Full trajectories can
make each 24-hour run about 190 MB for this catalog. Old runs remain available;
there is no automatic deletion. A failed run leaves an error record and never
replaces `results/latest.json`.

Load trajectories without pickle:

```python
import json
from pathlib import Path
import numpy as np

root = Path("results")
run = root / json.loads((root / "latest.json").read_text())["run_id"]
with np.load(run / "trajectories.npz", allow_pickle=False) as data:
    ids = data["norad_ids"]
    seconds = data["seconds_from_start"]
    r = data["position_teme_km"]       # [object, time, xyz]
    v = data["velocity_teme_km_s"]     # [object, time, xyz]
    errors = data["sgp4_error_codes"]  # [object, time]; 0 means success
    print(ids.shape, seconds.shape, r.shape, v.shape)
```

## Accuracy, precision, recall, and collision probability

The current download has no independently labeled outcomes. Observational accuracy, precision,
recall, F1, specificity, balanced accuracy, negative predictive value, false-positive
and false-negative rates, and MCC are saved as `null` with the reason. They are not
invented. The PINN's separately named SGP4-reference agreement scores are described
in step 3 above; they do not treat SGP4 as observational ground truth.

To calculate classification metrics later, supply independent **close-approach**
labels for the same threshold, object pairs, and exact analysis window. Include
negative pairs as well as positives. `labels_template.json` contains the required
metadata. Each entry in `pairs` must have `object1_id`, `object2_id`, and integer
`label` (0 or 1). The nonempty `reference_source` must identify the reference used.
Run with `--labels path/to/labels.json --start <matching-UTC-start>` and matching
settings/input. Metrics apply only to that labeled subset. Unknown, duplicate,
excluded, or shared-orbit pairs are rejected, as are mismatched windows/thresholds.

ROC AUC, PR AUC, Brier score, and collision probability remain unavailable because
there are no calibrated probability scores or orbital covariances/object radii.
Propagation success rate and numerical reference error are measured separately;
neither is an observational collision-prediction accuracy score.

Validation covers the published Vallado test case, OMM/TLE agreement, fast encounters
between time samples, brute-force comparison of the swept linear search, endpoint
minima, shared orbits, invalid propagation, metrics, and local dashboard routes:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

References: [SGP4 library](https://pypi.org/project/sgp4/),
[published verification outputs](https://github.com/brandon-rhodes/python-sgp4/blob/master/sgp4/tcppver.out),
[NASA conjunction probability guidance](https://nodis3.gsfc.nasa.gov/displayDir.cfm?Internal_ID=N_PR_8079_0001_&page_name=AppendixD).

## 1. Run the downloader

Python 3.10 or newer is required. This step uses only the Python standard library;
there are no packages to install and no API key to configure.

From PowerShell in `D:\PINN NEW`:

```powershell
.\run_fetch.ps1
```

The launcher uses installed Python or the Python bundled with this computer's
Codex installation. With Python on PATH, the portable equivalent is:

```powershell
python fetch_celestrak.py
```

By default, the script requests these four groups:

| Group | Contents |
| --- | --- |
| `stations` | Space stations and associated objects |
| `cosmos-2251-debris` | COSMOS 2251 debris |
| `iridium-33-debris` | IRIDIUM 33 debris |
| `fengyun-1c-debris` | FENGYUN 1C debris |

This is a starter selection, not a complete catalog of satellites or space debris.
Group membership is retained as provenance; it is not a general object-type label.

Choose another selection when needed, for example active satellites and one
debris group:

```powershell
.\run_fetch.ps1 --groups active cosmos-2251-debris
```

Use the same output directory between runs so the cache can be reused. The default
is `data` beside the Python script, even when launched from another directory.

```powershell
.\run_fetch.ps1 --offline
.\run_fetch.ps1 --help
```

Offline mode requires an existing cache for every selected group and explicitly
allows old downloads. It does not contact CelesTrak.

## 2. Inspect the saved data

Each successful run writes:

```text
data/
  cache/<group>.json                # Downloaded records and original download time
  latest.json                      # Pointer and metadata for the last complete run
  snapshots/<UTC timestamp>/
    raw/<group>.json               # Original record fields, reformatted as JSON
    orbital_data.json              # Merged records, one per NORAD catalog ID
    orbital_data.csv               # Same merged data in tabular form
    manifest.json                  # URLs, counts, epochs, download times, cache use
```

The newest epoch wins when an object appears in multiple groups. `SOURCE_GROUPS`
lists all its requested groups (a JSON list or semicolon-separated CSV value).
All source records remain in the raw files. Repeated cached runs may produce
identical snapshots; they are not new observations or independent training samples.

Load the latest merged data in Python:

```python
import json
from pathlib import Path

data_dir = Path("data")  # Run this example from the project directory.
latest = json.loads((data_dir / "latest.json").read_text(encoding="utf-8"))
snapshot = data_dir / latest["snapshot"]
records = json.loads((snapshot / "orbital_data.json").read_text(encoding="utf-8"))
print(f"Loaded {len(records)} objects")
print(records[0])
```

Important fields are:

| Field | Meaning / unit |
| --- | --- |
| `NORAD_CAT_ID` | Object catalog number; supports up to nine digits |
| `OBJECT_NAME`, `OBJECT_ID` | Name and international designator, when supplied |
| `EPOCH` | Time of the orbital element set, in UTC |
| `MEAN_MOTION` | Revolutions per day |
| `ECCENTRICITY` | Dimensionless orbital eccentricity |
| `INCLINATION` | Degrees |
| `RA_OF_ASC_NODE` | Right ascension of ascending node, degrees |
| `ARG_OF_PERICENTER` | Argument of pericenter, degrees |
| `MEAN_ANOMALY` | Mean anomaly, degrees |
| `BSTAR` | SGP4 drag parameter, inverse Earth radii |

The original mean-motion derivative fields and any additional API fields are
preserved. Apply the intended propagator's input conventions when using them later.

## 3. Understand freshness and failures

The endpoint is `https://celestrak.org/NORAD/elements/gp.php` with `GROUP` and
`FORMAT=JSON` query parameters. JSON uses OMM field names and supports catalog
numbers beyond the legacy TLE format's five-digit limit.

"Live" here means the latest published GP orbital elements when fetched, not a
stream of measured positions. `fetched_at_utc` records the download time; `EPOCH`
records the time of each orbit estimate. The manifest reports the epoch range and
the number of objects older than 72 hours, an informational threshold chosen for
this project, not an accuracy guarantee. Old epochs remain available for review.

Downloads are cached per group for two hours, following CelesTrak's usage policy.
There are no automatic retries. A failed request stops the run, and the previous
`latest.json` stays intact. Inspect the error before running again; HTTP 403/429
can indicate rate limiting, 404 an invalid query, and 5xx a server problem. A
failed online refresh never silently claims old data is fresh. Use `--offline`
explicitly to work with existing downloads. Run only one downloader at a time.

These GP elements use SGP4 mean-element conventions with Earth as the center,
TEME as the reference frame, and UTC as the time system. Those defaults are recorded
in the manifest because CelesTrak may omit them from JSON. Step 2 above now propagates
positions and velocities and screens close approaches in TEME. The independent
benchmark above also transforms the ESA reference into TEME. Calibrated collision
probabilities remain future work. Step 3 now implements
PINN physics losses and neural predictions with explicit evaluation limitations.
These downloads alone contain neither collision labels nor the uncertainty data
needed for a calibrated collision probability.

## 4. Verify the code

The tests use synthetic fixtures and mocked HTTP calls, so they do not request live
data or consume the API's download allowance:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The full test suite requires the base dependencies and PyTorch in the project venv.

Official references checked on 2026-09-10:

- [CelesTrak GP API and data formats](https://celestrak.org/NORAD/documentation/gp-data-formats.php)
- [Current orbital data groups](https://celestrak.org/NORAD/elements/)
- [CelesTrak usage policy](https://celestrak.org/usage-policy.php)
