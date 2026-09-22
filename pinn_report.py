"""Export figures and a readable report from actual training and inference arrays."""
import html
import os

import numpy as np

from run_analysis import ROOT


def make_report(folder,summary,history,seconds,position_errors,partitions,events):
    os.environ.setdefault("MPLCONFIGDIR",str(ROOT/".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures=folder/"figures";figures.mkdir(exist_ok=True)
    metrics=summary["metrics"]
    classification=metrics["sgp4_reference_pair_classification"]
    test=metrics["trajectory_agreement_vs_sgp4"]["test"]
    ablation=metrics["untrained_taylor_ablation_vs_sgp4"]["test"]
    fig,axes=plt.subplots(2,2,figsize=(13,9),constrained_layout=True)
    epochs=[row["step"] for row in history]
    axes[0,0].semilogy(epochs,[row["training_physics_loss"] for row in history],label="Training",color="#1a859c")
    axes[0,0].semilogy(epochs,[row["validation_physics_loss"] for row in history],label="Validation",color="#d0782a")
    axes[0,0].set(xlabel="Optimizer steps",ylabel="Scaled equation residual MSE",title="Actual PINN optimization")
    axes[0,0].legend()
    for name,color in [("train","#1a859c"),("validation","#7553b5"),("test","#d0782a")]:
        values=np.sqrt(np.mean(position_errors[partitions[name]]**2,axis=0))
        axes[0,1].plot(seconds/60,values,label=name,color=color)
    axes[0,1].set(xlabel="Minutes from initial condition",ylabel="Position vector RMSE (km)",title="Neural trajectories versus SGP4 reference")
    axes[0,1].legend()
    cm=classification["confusion_matrix"]
    matrix=np.array([[cm["true_negative"],cm["false_positive"]],[cm["false_negative"],cm["true_positive"]]])
    axes[1,0].imshow(np.log1p(matrix),cmap="Blues")
    for i in range(2):
        for j in range(2):axes[1,0].text(j,i,f"{matrix[i,j]:,}",ha="center",va="center",color="black" if matrix[i,j]<matrix.max()/10 else "white",fontsize=13)
    axes[1,0].set(xticks=[0,1],yticks=[0,1],xticklabels=["PINN negative","PINN positive"],
                  yticklabels=["SGP4 negative","SGP4 positive"],title="Pair agreement counts (not observed collisions)")
    axes[1,1].bar(["Untrained Taylor trial","Trained PINN"],
                  [ablation["position_vector_rmse_km"],test["position_vector_rmse_km"]],color=["#8997a6","#1a859c"])
    axes[1,1].set(ylabel="Held-out position vector RMSE versus SGP4 (km)",title="Measured effect of the learned correction")
    fig.suptitle("PINN experiment from CelesTrak initial conditions | SGP4 is a model reference",fontsize=14)
    fig.savefig(figures/"training_and_evaluation.png",dpi=170)
    plt.close(fig)
    fig,axis=plt.subplots(figsize=(10,4),constrained_layout=True)
    if events:
        axis.scatter([event["offset_seconds"]/60 for event in events],
                     [event["miss_distance_km"] for event in events],s=25,color="#1a859c",alpha=.75)
    else:axis.text(.5,.5,"No neural close approaches inside threshold",ha="center",transform=axis.transAxes)
    axis.axhline(summary["config"]["threshold_km"],linestyle="--",color="#d0782a",label="Screening threshold")
    axis.set(xlabel="Minutes from window start",ylabel="Neural miss distance (km)",title="Close approaches from trained PINN trajectories")
    axis.legend();fig.savefig(figures/"pinn_close_approaches.png",dpi=170);plt.close(fig)
    def show(value):return "Unavailable" if value is None else f"{value:.6g}" if isinstance(value,float) else str(value)
    metrics_table="\n".join(f"| {name} | {show(classification[name])} |" for name in
                            ("accuracy","precision","recall","f1_score","balanced_accuracy","specificity","matthews_correlation_coefficient"))
    comparison_note=("improved" if test["position_vector_rmse_km"]<ablation["position_vector_rmse_km"] else "did not improve")
    report=f"""# Actual PINN training and close-approach evaluation

Run: {summary['run_id']}
Forecast window: {summary['window_start_utc']} to {summary['window_end_utc']}

## What actually ran

- Hardware: {summary['hardware'].get('cpu','CPU')}; device {summary['hardware']['selected_device']}; {summary['hardware']['torch_threads']} PyTorch threads; float64.
- Model: {summary['training']['parameter_count']:,} trainable parameters; {summary['training']['optimizer_steps_executed']:,} optimizer steps executed.
- Training duration: {summary['training']['training_seconds']:.2f} seconds; peak sampled process memory: {summary['training']['peak_sampled_resident_memory_mb']:.1f} MB.
- Parameter-vector change from initialization: {summary['training']['parameter_l2_change']:.8g}.
- Input catalog: {summary['input_catalog_objects']:,} downloaded objects; modeled: {summary['modeled_objects']:,}; excluded: {summary['excluded_objects']:,}.
- Object split counts: {summary['split_counts']}. Identical modeled orbits stay in one split.
- Future SGP4 samples used for training: 0. Measured future tracks used for training: 0.

## Data and method

Real CelesTrak GP records were converted to initial Cartesian states with SGP4 at the run start.
The network learns a short-time flow map using automatic differentiation of the central-gravity + J2 equations.
Position and velocity initial conditions are enforced exactly. A Taylor trial solution supplies lower-order
motion, and a 3-layer tanh neural network learns higher-order corrections. It is rolled forward in
{summary['config']['segment_seconds']} second segments.

These are actual trained-model outputs. Collocation points enforce equations; they are not measured
trajectory observations. No future SGP4 trajectory was used to fit or select the model. Validation model
selection used equation residuals on held-out catalog initial states only. Future SGP4 states were
generated after training as a clearly identified reference for comparison.

## Measured neural outcomes

Neural close-approach episodes inside {summary['config']['threshold_km']} km: **{summary['pinn_close_approach_events']}**.
SGP4 reference close-approach episodes: **{summary['sgp4_reference_events']}**.
Closest neural approach: **{show(summary['closest_pinn_approach_km'])} km**.
Shared modeled-orbit pairs separated from the event register: {summary['shared_orbit_pairs']}.

Held-out test-object position vector RMSE against SGP4: **{test['position_vector_rmse_km']:.6g} km**.
Held-out final-time position vector RMSE: **{test['final_position_vector_rmse_km']:.6g} km**.
Held-out velocity vector RMSE: **{test['velocity_vector_rmse_km_s']:.6g} km/s**.
The learned correction {comparison_note} the untrained trial solution's held-out SGP4 agreement
(untrained position RMSE: {ablation['position_vector_rmse_km']:.6g} km).

## Classification agreement with SGP4

These metrics compare predicted pair flags within the same time window and distance threshold.
They are **not real-world collision accuracy**. These all-object pair scores include training and
validation objects. The pair universe includes negative pairs; it is not
restricted to the flagged events. Accuracy may be dominated by negatives.

| Metric | SGP4-reference agreement |
| --- | --- |
{metrics_table}

Confusion counts: {classification['confusion_matrix']}.
Total evaluated pair universe: {classification['pair_universe']:,}.
The test-only pair universe contains
{summary['metrics']['heldout_object_pair_classification_vs_sgp4']['reference_positive_pairs']} positive SGP4
reference pairs. Its precision and recall may therefore be undefined; the exact values are in metrics.json.
Held-out maximum position error was {test['position_vector_max_km']:.6g} km. The errors are substantial
relative to the screening threshold, so this experiment does not establish reliable collision prediction.

## Observational accuracy and collision probability

Observed collision accuracy, precision, recall, F1, ROC AUC, PR AUC, and Brier score are unavailable.
There are no observed event labels, measured future trajectories, or calibrated probability scores.

Collision probability status: **{summary['metrics']['collision_probability']['status']}**.
No covariance matrices or hard-body radii were invented. If real encounter-time covariances and justified
object radii are later supplied, the optional encounter-plane Gaussian calculation can be used within
its documented assumptions. Neural prediction errors alone are not treated as physical covariance.

## Contents of this folder

- model.pt: learned weights, architecture dimensions, best validation step, and seed.
- input_snapshot/, initial_conditions.npz, data_splits.json: exact input data and object partitions.
- hardware.json, config.json, training_proof.json: device selection and evidence of actual optimization.
- training_history.csv / .jsonl: real training and validation loss history.
- pinn_trajectories.npz: every neural position/velocity sample, full NORAD IDs and time offsets.
- sgp4_reference_trajectories.npz, sgp4_reference_events.json: separately generated comparison data.
- pinn_conjunctions.csv / .json: close approaches calculated directly from the neural model.
- per_object_evaluation.csv, reference_errors.npz: regression error data.
- metrics.csv / .json: all evaluation metrics and explicit missing-value reasons.
- pair_classification_vs_sgp4.csv, pair_universe.json: positive/negative comparison outcomes and the complete negative-universe definition.
- event_comparison_vs_sgp4.csv: matched pair encounter-time and miss-distance differences.
- pinn_candidate_audit.csv: refined candidate intervals; a/b index initial_conditions.npz norad_ids.
- shared_orbit_pairs.csv, excluded_objects.json: coverage limitations.
- figures/: actual loss curves, trajectory errors, confusion counts, ablation, and neural events.
- summary.json, status.json, run.log: run provenance, completion status, and source hashes.

## Limits

"""+"\n".join(f"- {note}" for note in summary["model_limitations"])+"\n"
    (folder/"report.md").write_text(report,encoding="utf-8")
    (folder/"report.html").write_text("<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>PINN experiment report</title><style>body{font:16px/1.6 system-ui;max-width:1100px;margin:40px auto;padding:0 20px;color:#172e3b}pre{white-space:pre-wrap;font:inherit}img{max-width:100%}</style><h1>PINN experiment report</h1><img src='figures/training_and_evaluation.png' alt='Actual training and evaluation curves'><img src='figures/pinn_close_approaches.png' alt='Neural close approaches'><pre>"+html.escape(report)+"</pre></html>",encoding="utf-8")
