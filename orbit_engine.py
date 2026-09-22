"""SGP4 propagation and continuous-interval conjunction screening in TEME.

Distance thresholds are research screening criteria, not collision probabilities.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import timedelta
import math

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.spatial import cKDTree
from sgp4 import omm
from sgp4.api import SGP4_ERRORS, Satrec, SatrecArray, WGS72, jday

from fetch_celestrak import parse_epoch, utc_text


@dataclass
class AnalysisConfig:
    horizon_hours: float = 24.0
    step_seconds: float = 60.0
    threshold_km: float = 5.0
    max_epoch_age_hours: float = 72.0
    include_stale: bool = False
    acceleration_pad_km_s2: float = 0.05

    def validate(self):
        limits = {
            "horizon_hours": (0.1, 48), "step_seconds": (10, 120),
            "threshold_km": (0.01, 50), "max_epoch_age_hours": (1, 720),
            "acceleration_pad_km_s2": (0.02, 1),
        }
        for name, (low, high) in limits.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be numeric")
            if not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{name} must be between {low} and {high}")
        if not isinstance(self.include_stale, bool):
            raise ValueError("include_stale must be true or false")
        return self

    def to_dict(self):
        return asdict(self)


def make_satellite(row):
    fields = dict(row)
    # sgp4's C++ wrapper still Alpha-5-encodes this metadata field. Catalog IDs
    # do not enter the equations: retain the real ID in our rows/array metadata
    # and use a neutral internal ID above the wrapper's current numeric limit.
    if int(fields["NORAD_CAT_ID"]) > 339999:
        fields["NORAD_CAT_ID"] = 0
    fields["EPOCH"] = parse_epoch(row["EPOCH"]).replace(tzinfo=None).isoformat(timespec="microseconds")
    fields.setdefault("OBJECT_ID", "")
    fields.setdefault("CLASSIFICATION_TYPE", "U")
    fields.setdefault("EPHEMERIS_TYPE", 0)
    fields.setdefault("ELEMENT_SET_NO", 0)
    fields.setdefault("REV_AT_EPOCH", 0)
    satellite = Satrec()
    omm.initialize(satellite, fields, WGS72)
    return satellite


def prepare_catalog(records):
    satellites, usable, excluded = [], [], []
    for row in records:
        try:
            satellites.append(make_satellite(row))
            usable.append(row)
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            excluded.append({"norad_id": row.get("NORAD_CAT_ID"), "reason": f"Initialization: {exc}"})
    if not usable:
        raise ValueError("No objects could be initialized for SGP4")
    return satellites, usable, excluded


def julian_dates(start, seconds):
    jd, fraction = jday(start.year, start.month, start.day, start.hour, start.minute,
                        start.second + start.microsecond / 1e6)
    parts = fraction + np.asarray(seconds, dtype=np.float64) / 86400.0
    days = np.floor(parts)
    return np.ascontiguousarray(jd + days), np.ascontiguousarray(parts - days)


def propagate(satellites, start, seconds):
    jd, fraction = julian_dates(start, seconds)
    errors, positions, velocities = SatrecArray(satellites).sgp4(jd, fraction)
    invalid = (errors != 0) | ~np.isfinite(positions).all(axis=2) | ~np.isfinite(velocities).all(axis=2)
    positions[invalid] = np.nan
    velocities[invalid] = np.nan
    return errors, positions, velocities


def interval_candidates(left, right, threshold, acceleration, duration):
    """Swept chord broad phase + exact linear relative closest approach.

    Each trajectory is padded by A*dt^2/8. This is a mathematical interpolation
    bound only if actual acceleration remains <= A. It is not an orbit covariance.
    """
    middle = (left + right) / 2
    half_length = np.linalg.norm(right - left, axis=1) / 2
    pad = acceleration * duration**2 / 8
    radii = half_length + pad
    tree = cKDTree(middle)
    pairs = tree.query_pairs(2 * float(radii.max()) + threshold, output_type="ndarray")
    if not len(pairs):
        return pairs.reshape(0, 2), np.empty(0), 0
    broad_count = len(pairs)
    a, b = pairs.T
    possible = np.linalg.norm(middle[a] - middle[b], axis=1) <= radii[a] + radii[b] + threshold
    pairs = pairs[possible]
    a, b = pairs.T
    relative = left[a] - left[b]
    change = (right[a] - left[a]) - (right[b] - left[b])
    denominator = np.einsum("ij,ij->i", change, change)
    fraction = np.divide(-np.einsum("ij,ij->i", relative, change), denominator,
                         out=np.zeros(len(pairs)), where=denominator > 1e-18)
    fraction = np.clip(fraction, 0, 1)
    distance = np.linalg.norm(relative + fraction[:, None] * change, axis=1)
    keep = distance <= threshold + 2 * pad
    return pairs[keep], fraction[keep], broad_count


def pair_state(sat_a, sat_b, start, seconds):
    jd, fraction = julian_dates(start, [seconds])
    ea, ra, va = sat_a.sgp4(float(jd[0]), float(fraction[0]))
    eb, rb, vb = sat_b.sgp4(float(jd[0]), float(fraction[0]))
    if ea or eb or not np.isfinite([*ra, *rb, *va, *vb]).all():
        raise ValueError(f"SGP4 refinement error: {ea}, {eb}")
    return np.asarray(ra), np.asarray(rb), np.asarray(va), np.asarray(vb)


def refine_interval(sat_a, sat_b, start, lo, hi):
    # Optimize a small local offset to retain timing precision for later intervals.
    def squared_distance(offset):
        ra, rb, _, _ = pair_state(sat_a, sat_b, start, lo + offset)
        return float(np.dot(ra - rb, ra - rb))
    result = minimize_scalar(squared_distance, bounds=(0, hi - lo), method="bounded",
                             options={"xatol": 0.001, "maxiter": 80})
    if not result.success:
        raise ValueError("Closest-approach minimization did not converge")
    choices = [(0.0, squared_distance(0)), (hi - lo, squared_distance(hi - lo)),
               (float(result.x), float(result.fun))]
    offset, distance2 = min(choices, key=lambda item: item[1])
    return lo + offset, math.sqrt(max(0, distance2))


def screen(satellites, rows, positions, errors, start, seconds, config, progress=lambda *args: None):
    valid = (errors == 0).all(axis=1) & np.isfinite(positions).all(axis=(1, 2))
    epoch_age = np.array([(start - parse_epoch(row["EPOCH"])).total_seconds() / 3600 for row in rows])
    stale = np.abs(epoch_age) > config.max_epoch_age_hours
    eligible = valid & (np.ones(len(rows), dtype=bool) if config.include_stale else ~stale)
    indices = np.flatnonzero(eligible)
    exclusions = []
    for index, row in enumerate(rows):
        if not eligible[index]:
            codes = sorted(set(int(code) for code in errors[index] if code))
            reason = ("; ".join(SGP4_ERRORS.get(code, str(code)) for code in codes)
                      if not valid[index] else "Epoch outside configured age limit")
            exclusions.append({"norad_id": int(row["NORAD_CAT_ID"]), "reason": reason or "Nonfinite state",
                               "epoch_age_hours": float(epoch_age[index]), "error_codes": codes})
    if len(indices) < 2:
        raise ValueError("Fewer than two eligible objects remain; inspect epochs or adjust the age limit")
    # Different catalog IDs can carry exactly the same modeled orbit (for example,
    # station modules). GP data cannot resolve a conjunction between these entries.
    signatures = {}
    for index in indices:
        row = rows[index]
        key = (utc_text(parse_epoch(row["EPOCH"])), *(float(row[field]) for field in
               ("MEAN_MOTION", "ECCENTRICITY", "INCLINATION", "RA_OF_ASC_NODE",
                "ARG_OF_PERICENTER", "MEAN_ANOMALY", "BSTAR")))
        signatures.setdefault(key, []).append(int(index))
    shared_indices = set()
    shared_pairs = []
    for members in signatures.values():
        for offset, a in enumerate(members):
            for b in members[offset + 1:]:
                shared_indices.add((a, b))
                shared_pairs.append({"object1_id": int(rows[a]["NORAD_CAT_ID"]),
                                     "object2_id": int(rows[b]["NORAD_CAT_ID"]),
                                     "object1_name": rows[a].get("OBJECT_NAME", ""),
                                     "object2_name": rows[b].get("OBJECT_NAME", ""),
                                     "reason": "Identical modeled orbital elements; no independent separation can be resolved"})
    candidates = {}
    broad_count = 0
    for index in range(len(seconds) - 1):
        duration = float(seconds[index + 1] - seconds[index])
        pairs, _, count = interval_candidates(positions[indices, index], positions[indices, index + 1],
                                              config.threshold_km, config.acceleration_pad_km_s2, duration)
        broad_count += count
        for a, b in pairs:
            key = (int(indices[a]), int(indices[b]))
            if key in shared_indices:
                continue
            candidates.setdefault(key, []).append(index)
        if index % 120 == 0:
            progress(20 + 40 * index / (len(seconds) - 1),
                     f"Screening interval {index + 1:,} / {len(seconds) - 1:,}")
    events, audit, refinement_errors = [], [], []
    candidate_intervals = sum(len(value) for value in candidates.values())
    done = 0
    for (a, b), intervals in candidates.items():
        # Keep repeated encounters when separated by a non-candidate interval.
        episodes = np.split(intervals, np.flatnonzero(np.diff(intervals) > 1) + 1)
        for episode in episodes:
            best = (None, float("inf"))
            failed = False
            for index in episode:
                try:
                    tca, distance = refine_interval(satellites[a], satellites[b], start,
                                                    float(seconds[index]), float(seconds[index + 1]))
                    if distance < best[1]:
                        best = (tca, distance)
                except ValueError as exc:
                    failed = True
                    refinement_errors.append({"object1_id": int(rows[a]["NORAD_CAT_ID"]),
                                              "object2_id": int(rows[b]["NORAD_CAT_ID"]),
                                              "interval": int(index), "reason": str(exc)})
                done += 1
                if done % 3000 == 0:
                    progress(60 + 25 * done / max(1, candidate_intervals),
                             f"Refining candidate interval {done:,} / {candidate_intervals:,}")
            tca, distance = best
            candidate = {"object1_id": int(rows[a]["NORAD_CAT_ID"]), "object2_id": int(rows[b]["NORAD_CAT_ID"]),
                         "bracket_start_utc": utc_text(start + timedelta(seconds=float(seconds[episode[0]]))),
                         "bracket_end_utc": utc_text(start + timedelta(seconds=float(seconds[episode[-1] + 1]))),
                         "refinement_failed": failed, "minimum_distance_km": distance if tca is not None else None}
            audit.append(candidate)
            if tca is None or distance > config.threshold_km:
                continue
            ra, rb, va, vb = pair_state(satellites[a], satellites[b], start, tca)
            end = float(seconds[-1])
            curve_seconds = np.linspace(max(0, tca - 120), min(end, tca + 120), 81)
            curve = []
            for second in curve_seconds:
                try:
                    p1, p2, _, _ = pair_state(satellites[a], satellites[b], start, float(second))
                    curve.append([round(float(second - tca), 3), float(np.linalg.norm(p1 - p2))])
                except ValueError:
                    curve.append([round(float(second - tca), 3), None])
            events.append({
                **candidate, "object1_name": rows[a].get("OBJECT_NAME", str(rows[a]["NORAD_CAT_ID"])),
                "object2_name": rows[b].get("OBJECT_NAME", str(rows[b]["NORAD_CAT_ID"])),
                "tca_utc": utc_text(start + timedelta(seconds=tca)), "tca_offset_seconds": tca,
                "miss_distance_km": distance, "relative_speed_km_s": float(np.linalg.norm(va - vb)),
                "object1_epoch_age_hours_at_tca": float(epoch_age[a] + tca / 3600),
                "object2_epoch_age_hours_at_tca": float(epoch_age[b] + tca / 3600),
                "stale_at_start": bool(stale[a] or stale[b]),
                "at_window_boundary": tca < 0.01 or end - tca < 0.01,
                "position1_teme_km": ra.tolist(), "position2_teme_km": rb.tolist(),
                "distance_curve": curve,
                "collision_probability": None,
            })
    events.sort(key=lambda event: event["miss_distance_km"])
    for index, event in enumerate(events):
        event["event_id"] = f"CA-{index + 1:05d}"
    stats = {"screened_objects": len(indices), "possible_pairs": len(indices) * (len(indices) - 1) // 2 - len(shared_pairs),
             "shared_orbit_pairs_excluded": len(shared_pairs), "shared_orbit_pairs": shared_pairs,
             "broad_phase_pair_intervals": broad_count, "candidate_pairs": len(candidates),
             "candidate_intervals": candidate_intervals, "candidate_episodes": len(audit),
             "refinement_failures": len(refinement_errors), "conjunction_events": len(events),
             "distinct_flagged_pairs": len({(event["object1_id"], event["object2_id"]) for event in events}),
             "stale_objects_at_start": int(stale.sum()), "eligible_ids": [int(rows[i]["NORAD_CAT_ID"]) for i in indices]}
    return events, audit, exclusions, refinement_errors, stats


def reference_validation():
    """Published Vallado verification case 00005 at epoch (tcppver.out).

    Agreement with a reference implementation is not accuracy against observations.
    """
    sat = Satrec.twoline2rv(
        "1 00005U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753",
        "2 00005  34.2682 348.7242 1859667 331.7664  19.3264 10.82419157413667", WGS72)
    error, position, velocity = sat.sgp4(sat.jdsatepoch, sat.jdsatepochF)
    reference_r = np.array([7022.46529266, -1400.08296755, 0.03995155])
    reference_v = np.array([1.893841015, 6.405893759, 4.534807250])
    position_error = float(np.linalg.norm(np.asarray(position) - reference_r))
    velocity_error = float(np.linalg.norm(np.asarray(velocity) - reference_v))
    return {"kind": "implementation_reference_agreement", "reference": "Vallado case 00005, epoch, tcppver.out",
            "source": "https://github.com/brandon-rhodes/python-sgp4/blob/master/sgp4/tcppver.out",
            "position_vector_error_km": position_error, "velocity_vector_error_km_s": velocity_error,
            "passed": bool(error == 0 and position_error < 1e-6 and velocity_error < 1e-8),
            "note": "One numerical reference case; not measured orbital or collision accuracy."}
