"""Measured model/reference agreement, never substituted for observational truth."""
import math
import numpy as np

from pinn_model import LENGTH_KM, SPEED_KM_S


def trajectory_metrics(prediction, reference_r, reference_v, partitions):
    position_error=np.linalg.norm(prediction[:,:,:3]*LENGTH_KM-reference_r,axis=-1)
    velocity_error=np.linalg.norm(prediction[:,:,3:]*SPEED_KM_S-reference_v,axis=-1)
    metrics={}
    for name,indices in {"all_objects":np.arange(len(prediction)),**partitions}.items():
        pe=position_error[indices,1:]
        ve=velocity_error[indices,1:]
        metrics[name]={"objects":len(indices),"evaluated_future_states":int(pe.size),
                       "position_vector_rmse_km":float(np.sqrt(np.mean(pe**2))),
                       "position_vector_mae_km":float(np.mean(pe)),
                       "position_vector_p95_km":float(np.quantile(pe,.95)),
                       "position_vector_max_km":float(np.max(pe)),
                       "final_position_vector_rmse_km":float(np.sqrt(np.mean(position_error[indices,-1]**2))),
                       "velocity_vector_rmse_km_s":float(np.sqrt(np.mean(ve**2))),
                       "velocity_vector_mae_km_s":float(np.mean(ve)),
                       "reference":"SGP4 predictions from the same catalog elements; not observations"}
    return metrics,position_error,velocity_error


def pair_classification(pinn_events, reference_events, ids, shared_pairs):
    ids=set(int(x) for x in ids)
    excluded={tuple(sorted((p["object1_id"],p["object2_id"]))) for p in shared_pairs
              if p["object1_id"] in ids and p["object2_id"] in ids}
    def pairs(events):
        return {tuple(sorted((e["object1_id"],e["object2_id"]))) for e in events
                if e["object1_id"] in ids and e["object2_id"] in ids} - excluded
    predicted,reference=pairs(pinn_events),pairs(reference_events)
    tp=len(predicted&reference);fp=len(predicted-reference);fn=len(reference-predicted)
    universe=len(ids)*(len(ids)-1)//2-len(excluded)
    tn=universe-tp-fp-fn
    def ratio(a,b):return a/b if b else None
    precision,recall,specificity=ratio(tp,tp+fp),ratio(tp,tp+fn),ratio(tn,tn+fp)
    metrics={"interpretation":"Agreement with SGP4 pair flags in the SAME future window, not observed collision accuracy",
             "pair_universe":universe,"reference_positive_pairs":len(reference),"pinn_positive_pairs":len(predicted),
             "reference_prevalence":ratio(len(reference),universe),
             "confusion_matrix":{"true_positive":tp,"false_positive":fp,"false_negative":fn,"true_negative":tn},
             "accuracy":ratio(tp+tn,universe),"precision":precision,"recall":recall,"specificity":specificity,
             "f1_score":ratio(2*tp,2*tp+fp+fn),
             "balanced_accuracy":(recall+specificity)/2 if recall is not None and specificity is not None else None,
             "negative_predictive_value":ratio(tn,tn+fn),"false_positive_rate":ratio(fp,fp+tn),
             "false_negative_rate":ratio(fn,fn+tp),
             "matthews_correlation_coefficient":ratio(tp*tn-fp*fn,math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))),
             "roc_auc":None,"pr_auc":None,"brier_score":None,
             "note":"Accuracy can be dominated by millions of negative pairs; inspect precision, recall, and counts."}
    rows=[{"object1_id":a,"object2_id":b,"pinn_positive":int((a,b) in predicted),
           "sgp4_reference_positive":int((a,b) in reference),
           "outcome":"TP" if (a,b) in predicted&reference else "FP" if (a,b) in predicted else "FN"}
          for a,b in sorted(predicted|reference)]
    return metrics,rows


def event_comparison(pinn_events, reference_events):
    by_pair={}
    for event in reference_events:
        key=tuple(sorted((event["object1_id"],event["object2_id"])))
        by_pair.setdefault(key,[]).append(event)
    rows=[]
    for event in pinn_events:
        key=tuple(sorted((event["object1_id"],event["object2_id"])))
        candidates=by_pair.get(key,[])
        reference=min(candidates,key=lambda item:abs(item["tca_offset_seconds"]-event["offset_seconds"])) if candidates else None
        rows.append({"event_id":event["event_id"],"object1_id":event["object1_id"],"object2_id":event["object2_id"],
                     "pinn_miss_distance_km":event["miss_distance_km"],"pinn_tca_utc":event["tca_utc"],
                     "sgp4_miss_distance_km":reference["miss_distance_km"] if reference else None,
                     "sgp4_tca_utc":reference["tca_utc"] if reference else None,
                     "miss_distance_difference_km":event["miss_distance_km"]-reference["miss_distance_km"] if reference else None,
                     "tca_difference_seconds":event["offset_seconds"]-reference["tca_offset_seconds"] if reference else None})
    return rows
