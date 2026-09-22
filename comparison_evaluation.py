"""Offline SGP4/PINN comparison metrics and one-to-one encounter matching."""
from collections import defaultdict

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from pinn_model import (EARTH_RADIUS_KM, J2, LENGTH_KM, MU_KM3_S2,
                        TIME_SECONDS, acceleration)


def match_encounters(pinn_events, sgp4_events, tolerance_seconds=120):
    """Maximize matched same-pair events within tolerance, then minimize time error."""
    if not np.isfinite(tolerance_seconds) or tolerance_seconds <= 0:
        raise ValueError("Matching tolerance must be positive")
    groups = defaultdict(lambda: [[], []])
    for side, events in enumerate((pinn_events, sgp4_events)):
        for event in events:
            pair = tuple(sorted((event["object1_id"], event["object2_id"])))
            groups[pair][side].append(event)
    rows = []
    for pair, (neural, reference) in sorted(groups.items()):
        matches = []
        if neural and reference:
            delta = np.abs(np.array([e["offset_seconds"] for e in neural])[:, None]
                           - np.array([e["tca_offset_seconds"] for e in reference])[None, :])
            # One invalid assignment must cost more than all valid assignments combined.
            penalty = (max(len(neural), len(reference)) + 1) * (tolerance_seconds + 1)
            a, b = linear_sum_assignment(np.where(delta <= tolerance_seconds, delta, penalty))
            matches = [(int(i), int(j)) for i, j in zip(a, b) if delta[i, j] <= tolerance_seconds]
        for i, j in matches:
            n, s = neural[i], reference[j]
            rows.append({"object1_id":pair[0], "object2_id":pair[1], "status":"matched",
                         "pinn_event_id":n["event_id"], "sgp4_event_id":s["event_id"],
                         "pinn_tca_utc":n["tca_utc"], "sgp4_tca_utc":s["tca_utc"],
                         "pinn_miss_distance_km":n["miss_distance_km"], "sgp4_miss_distance_km":s["miss_distance_km"],
                         "tca_difference_seconds":n["offset_seconds"]-s["tca_offset_seconds"],
                         "miss_distance_difference_km":n["miss_distance_km"]-s["miss_distance_km"],
                         "relative_speed_difference_km_s":n["relative_speed_km_s"]-s["relative_speed_km_s"]})
        used_n, used_s = {i for i, _ in matches}, {j for _, j in matches}
        for side, events, used in (("pinn", neural, used_n), ("sgp4", reference, used_s)):
            for index, event in enumerate(events):
                if index not in used:
                    rows.append({"object1_id":pair[0], "object2_id":pair[1], "status":side+"_only",
                                 side+"_event_id":event["event_id"], side+"_tca_utc":event["tca_utc"],
                                 side+"_miss_distance_km":event["miss_distance_km"]})
    matched = [row for row in rows if row["status"] == "matched"]
    tp = len(matched)
    fp, fn = sum(row["status"] == "pinn_only" for row in rows), sum(row["status"] == "sgp4_only" for row in rows)
    metrics = {"matched_events":tp, "pinn_only_events":fp, "sgp4_only_events":fn,
               "matching_tolerance_seconds":tolerance_seconds,
               "precision":tp/(tp+fp) if tp+fp else None, "recall":tp/(tp+fn) if tp+fn else None,
               "f1_score":2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
               "interpretation":"One-to-one same-pair episode agreement, not observed events; no event TN universe is defined."}
    for field in ("tca_difference_seconds", "miss_distance_difference_km", "relative_speed_difference_km_s"):
        values = np.array([row[field] for row in matched])
        metrics[field+"_mae"] = float(np.mean(np.abs(values))) if values.size else None
        metrics[field+"_rmse"] = float(np.sqrt(np.mean(values**2))) if values.size else None
        metrics[field+"_max_absolute"] = float(np.max(np.abs(values))) if values.size else None
    return metrics, rows


def trajectory_details(pinn_r, pinn_v, sgp4_r, sgp4_v, ids, rows, partitions, seconds):
    dr, dv = pinn_r-sgp4_r, pinn_v-sgp4_v
    pe, ve = np.linalg.norm(dr, axis=-1), np.linalg.norm(dv, axis=-1)
    radial = sgp4_r / np.linalg.norm(sgp4_r, axis=-1, keepdims=True)
    normal = np.cross(sgp4_r, sgp4_v)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    transverse = np.cross(normal, radial)
    rtn = np.stack([np.sum(dr*axis, axis=-1) for axis in (radial, transverse, normal)], axis=-1)
    extra, time_rows, object_rows = {}, [], []
    membership = {int(index):name for name, indices in partitions.items() for index in indices}
    for name, indices in {"all_objects":np.arange(len(ids)), **partitions}.items():
        extra[name] = {"position_axis_rmse_km":np.sqrt(np.mean(dr[indices, 1:]**2, axis=(0, 1))).tolist(),
                       "position_axis_bias_km":np.mean(dr[indices, 1:], axis=(0, 1)).tolist(),
                       "velocity_axis_rmse_km_s":np.sqrt(np.mean(dv[indices, 1:]**2, axis=(0, 1))).tolist(),
                       "position_rtn_rmse_km":np.sqrt(np.mean(rtn[indices, 1:]**2, axis=(0, 1))).tolist(),
                       "velocity_vector_p95_km_s":float(np.quantile(ve[indices, 1:], .95)),
                       "velocity_vector_max_km_s":float(np.max(ve[indices, 1:])),
                       "position_axis_order":["TEME x", "TEME y", "TEME z"],
                       "rtn_axis_order":["radial", "transverse", "normal"]}
        for t, second in enumerate(seconds):
            time_rows.append({"split":name, "seconds_from_start":float(second),
                              "position_vector_rmse_km":float(np.sqrt(np.mean(pe[indices, t]**2))),
                              "position_vector_mae_km":float(np.mean(pe[indices, t])),
                              "position_vector_p95_km":float(np.quantile(pe[indices, t], .95)),
                              "position_vector_max_km":float(np.max(pe[indices, t])),
                              "velocity_vector_rmse_km_s":float(np.sqrt(np.mean(ve[indices, t]**2)))})
    for i, norad_id in enumerate(ids):
        object_rows.append({"norad_id":int(norad_id), "name":rows[i].get("OBJECT_NAME", ""),
                            "split":membership[i], "epoch_utc":rows[i]["EPOCH"],
                            "position_vector_rmse_km":float(np.sqrt(np.mean(pe[i, 1:]**2))),
                            "position_vector_mae_km":float(np.mean(pe[i, 1:])),
                            "maximum_position_error_km":float(np.max(pe[i, 1:])),
                            "final_position_error_km":float(pe[i, -1]),
                            "velocity_vector_rmse_km_s":float(np.sqrt(np.mean(ve[i, 1:]**2)))})
    return extra, time_rows, object_rows, pe, ve, rtn


def physics_diagnostics(position, velocity, seconds):
    """Same central+J2 consistency diagnostic for both models, not observed accuracy."""
    r = np.linalg.norm(position, axis=-1)
    p2 = .5*(3*(position[:, :, 2]/r)**2-1)
    potential = -MU_KM3_S2/r*(1-J2*(EARTH_RADIUS_KM/r)**2*p2)
    energy = .5*np.sum(velocity**2, axis=-1)+potential
    relative_energy = np.abs((energy-energy[:, :1])/energy[:, :1])
    hz = np.cross(position, velocity)[:, :, 2]
    delta_hz = np.abs(hz-hz[:, :1])
    with torch.no_grad():
        a = acceleration(torch.as_tensor(position/LENGTH_KM, dtype=torch.float64)).numpy()*LENGTH_KM/TIME_SECONDS**2
    # Interior centered differences only; the finite step also contributes error.
    kinematic = np.gradient(position, seconds, axis=1, edge_order=2)-velocity
    dynamic = np.gradient(velocity, seconds, axis=1, edge_order=2)-a
    return {"j2_energy_relative_drift_p95":float(np.quantile(relative_energy[:, 1:], .95)),
            "j2_energy_relative_drift_max":float(np.max(relative_energy[:, 1:])),
            "axial_angular_momentum_drift_p95_km2_s":float(np.quantile(delta_hz[:, 1:], .95)),
            "kinematic_residual_vector_rms_km_s":float(np.sqrt(np.mean(np.sum(kinematic[:, 1:-1]**2, axis=-1)))),
            "central_j2_acceleration_residual_vector_rms_km_s2":float(np.sqrt(np.mean(np.sum(dynamic[:, 1:-1]**2, axis=-1)))),
            "interpretation":"Consistency with central gravity + J2 in quasi-inertial TEME, not observational error. Centered differences have discretization error. SGP4 includes different perturbations, so these diagnostics do not rank real-world accuracy."}


def flatten_metrics(value, prefix=""):
    output = []
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            output.extend(flatten_metrics(item, name))
        else:
            output.append({"metric":name, "value":item, "status":"unavailable" if item is None else "recorded"})
    return output
