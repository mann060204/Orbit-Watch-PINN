"""Evaluate a distance-threshold classifier only against explicit reference labels."""
import math

from fetch_celestrak import parse_epoch


CLASSIFICATION_NAMES = ("accuracy", "precision", "recall", "f1_score", "specificity",
                        "balanced_accuracy", "negative_predictive_value", "false_positive_rate",
                        "false_negative_rate", "matthews_correlation_coefficient")


def classification_metrics(events, summary, labels=None):
    result = {"status": "not_evaluable", "target": "close_approach_within_threshold",
              "reason": "No independent labeled pair outcomes supplied. SGP4 output is not ground truth.",
              "evaluated_pairs": 0, "confusion_matrix": None,
              **{name: None for name in CLASSIFICATION_NAMES},
              "roc_auc": None, "pr_auc": None, "brier_score": None,
              "probability_metrics_reason": "No calibrated probability scores or covariance data are available."}
    if labels is None:
        return result
    if labels.get("target") != result["target"]:
        raise ValueError("Reference target must be close_approach_within_threshold")
    if float(labels["threshold_km"]) != summary["config"]["threshold_km"]:
        raise ValueError("Reference distance threshold does not match this run")
    for key in ("window_start_utc", "window_end_utc"):
        if parse_epoch(labels[key]) != parse_epoch(summary[key]):
            raise ValueError(f"Reference {key} does not match this run")
    if not isinstance(labels.get("reference_source"), str) or not labels["reference_source"].strip():
        raise ValueError("Provide a named independent reference_source")
    pairs = labels.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("Reference pairs must contain both explicit IDs and a 0/1 label")
    eligible = set(summary["screening"]["eligible_ids"])
    excluded_pairs = {tuple(sorted((pair["object1_id"], pair["object2_id"])))
                      for pair in summary["screening"].get("shared_orbit_pairs", [])}
    if summary["screening"].get("refinement_failures", 0):
        raise ValueError("Classification evaluation requires a run without unresolved refinement failures")
    predicted = {tuple(sorted((e["object1_id"], e["object2_id"]))) for e in events}
    counts = {"true_positive": 0, "false_positive": 0, "true_negative": 0, "false_negative": 0}
    seen = set()
    for row in pairs:
        ids = (row["object1_id"], row["object2_id"])
        if any(isinstance(value, bool) or not isinstance(value, int) for value in ids):
            raise ValueError("Reference object IDs must be integers")
        a, b = sorted(ids)
        actual = row["label"]
        if isinstance(actual, bool) or not isinstance(actual, int) or actual not in (0, 1):
            raise ValueError("Reference labels must be integer 0 or 1")
        if a == b or a not in eligible or b not in eligible or (a, b) in seen or (a, b) in excluded_pairs:
            raise ValueError("Reference contains duplicate, identical, unknown, or excluded pairs")
        seen.add((a, b))
        prediction = int((a, b) in predicted)
        key = {(1, 1): "true_positive", (0, 1): "false_positive",
               (0, 0): "true_negative", (1, 0): "false_negative"}[(actual, prediction)]
        counts[key] += 1
    tp, fp, tn, fn = (counts[key] for key in ("true_positive", "false_positive", "true_negative", "false_negative"))
    def ratio(numerator, denominator):
        return numerator / denominator if denominator else None
    recall, specificity = ratio(tp, tp + fn), ratio(tn, tn + fp)
    return {**result, "status": "evaluated_on_supplied_reference_pairs",
            "reason": "Metrics cover the explicitly labeled subset only; not observed collision accuracy.",
            "reference_source": labels["reference_source"], "evaluated_pairs": len(seen),
            "labeled_fraction_of_screened_pairs": len(seen) / summary["screening"]["possible_pairs"],
            "confusion_matrix": counts, "accuracy": (tp + tn) / len(seen),
            "precision": ratio(tp, tp + fp), "recall": recall, "f1_score": ratio(2 * tp, 2 * tp + fp + fn),
            "specificity": specificity,
            "balanced_accuracy": (recall + specificity) / 2 if recall is not None and specificity is not None else None,
            "negative_predictive_value": ratio(tn, tn + fn), "false_positive_rate": ratio(fp, fp + tn),
            "false_negative_rate": ratio(fn, fn + tp),
            "matthews_correlation_coefficient": ratio(tp * tn - fp * fn, math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))}
