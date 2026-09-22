"""Read-only, source-attributed context for a selected catalog object.

History is bundled mission/family knowledge plus actual saved GP snapshots.
No benchmark results or external network requests enter this dashboard context.
"""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import json
import math
from pathlib import Path
import re

import numpy as np

from fetch_celestrak import parse_epoch, utc_now, utc_text
from orbit_engine import propagate


MAX_SNAPSHOTS = 32
MAX_HISTORY_EPOCHS = 8
MAX_ENCOUNTERS = 5
MAX_SNAPSHOT_BYTES = 12 * 1024 * 1024
EARTH_RADIUS_KM = 6378.137  # Same geocentric display convention as the globe.


def _number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError, OverflowError):
        return None


def _utc(value):
    try:
        return utc_text(parse_epoch(value))
    except (ValueError, TypeError, AttributeError):
        return None


@lru_cache(maxsize=1)
def _knowledge():
    return json.loads(Path(__file__).with_name("object_knowledge.json").read_text(encoding="utf-8"))


def _snapshot_signature(root):
    folder = Path(root) / "data" / "snapshots"
    if not folder.is_dir():
        return ()
    snapshots = sorted((p for p in folder.iterdir() if p.is_dir() and
                        re.fullmatch(r"\d{8}T\d{6}_\d{6}Z", p.name)), reverse=True)[:MAX_SNAPSHOTS]
    signature = []
    for snapshot in snapshots:
        files = []
        for name in ("orbital_data.json", "manifest.json"):
            path = snapshot / name
            try:
                stat = path.stat()
                files.append((str(path.resolve()), stat.st_mtime_ns, stat.st_size))
            except OSError:
                files.append((str(path.resolve()), None, None))
        signature.append(tuple(files))
    return tuple(signature)


@lru_cache(maxsize=2)
def _snapshot_index(signature):
    """Cache compact histories for all objects; invalidate on saved-file changes."""
    histories, skipped = {}, 0
    for data_file, manifest_file in signature:
        try:
            if data_file[2] is None or data_file[2] > MAX_SNAPSHOT_BYTES:
                raise ValueError("Snapshot unavailable or oversized")
            if manifest_file[2] is None or manifest_file[2] > MAX_SNAPSHOT_BYTES:
                raise ValueError("Manifest unavailable or oversized")
            rows = json.loads(Path(data_file[0]).read_text(encoding="utf-8"))
            manifest = json.loads(Path(manifest_file[0]).read_text(encoding="utf-8"))
            if not isinstance(rows, list) or not isinstance(manifest, dict):
                raise ValueError("Invalid snapshot schema")
            fetched = {s.get("group"): _utc(s.get("fetched_at_utc"))
                       for s in manifest.get("sources", []) if isinstance(s, dict)}
            for row in rows:
                if not isinstance(row, dict):
                    continue
                try:
                    object_id = int(row["NORAD_CAT_ID"])
                except (KeyError, ValueError, TypeError, OverflowError):
                    continue
                epoch = _utc(row.get("EPOCH"))
                if epoch is None:
                    continue
                saved = histories.setdefault(object_id, {})
                if epoch in saved:
                    continue
                fetch_times = [fetched[g] for g in row.get("SOURCE_GROUPS", []) if fetched.get(g)]
                saved[epoch] = {
                    "epoch_utc": epoch,
                    "fetched_at_utc": max(fetch_times) if fetch_times else None,
                    "mean_motion_rev_day": _number(row.get("MEAN_MOTION")),
                    "inclination_deg": _number(row.get("INCLINATION")),
                }
        except (OSError, ValueError, TypeError, AttributeError):
            skipped += 1
    return {key: [rows[epoch] for epoch in sorted(rows, reverse=True)[:MAX_HISTORY_EPOCHS]]
            for key, rows in histories.items()}, skipped


def _history(row):
    knowledge = _knowledge()
    known = knowledge["objects"].get(str(row["NORAD_CAT_ID"]))
    if known:
        return deepcopy(known)
    groups = row.get("SOURCE_GROUPS", [])
    for group, family in knowledge["families"].items():
        if group in groups:
            result = deepcopy(family)
            if int(row["NORAD_CAT_ID"]) == family["parent_id"]:
                result["category"] = "Cataloged satellite remnant"
                result["history_scope"] = ("History of the original satellite and its breakup. "
                    "This catalog entry retains the original designation; it does not imply that the spacecraft is intact or operating.")
            else:
                result["category"] = "Debris family object"
            return result
    return {
        "category": "Stations catalog object" if "stations" in groups else "Catalog object",
        "history_summary": "No source-verified mission biography is bundled for this catalog object yet. Its current orbital elements and saved catalog updates are available below.",
        "history_scope": "Catalog identity only. Membership in the stations group does not by itself establish the object's mission, crew, docking state, or operational status.",
        "history_events": [], "source_ids": [],
    }


def _encounters(service, norad_id):
    with service.lock:
        summary = service.summary or {}
        selected = [event for event in service.events if norad_id in
                    (event.get("object1_id"), event.get("object2_id"))]
        screening = summary.get("screening", {})
        eligible = screening.get("eligible_ids")
        screened = norad_id in eligible if isinstance(eligible, list) else None
        shared_pairs = sum(norad_id in (p.get("object1_id"), p.get("object2_id"))
                           for p in screening.get("shared_orbit_pairs", []))
        events = []
        for event in sorted(selected, key=lambda e: _number(e.get("miss_distance_km"))
                            if _number(e.get("miss_distance_km")) is not None else math.inf)[:MAX_ENCOUNTERS]:
            other = 2 if event.get("object1_id") == norad_id else 1
            events.append({
                "other_id": event.get(f"object{other}_id"),
                "other_name": event.get(f"object{other}_name"),
                "tca_utc": event.get("tca_utc"),
                "miss_distance_km": _number(event.get("miss_distance_km")),
                "relative_speed_km_s": _number(event.get("relative_speed_km_s")),
                "collision_probability": None,
                "stale_at_start": event.get("stale_at_start"),
                "refinement_failed": event.get("refinement_failed", False),
            })
        note = "Saved SGP4 close-approach estimates, ordered by miss distance; not observed collisions. Collision probability is unavailable without covariance and object-size information."
        if not summary:
            note = "No completed SGP4 screening is available."
        elif screened is False:
            note = "This object was not included in the latest SGP4 screening. An empty list does not establish that it is safe."
        elif not selected:
            note = "No saved event for this object within this run's catalog, time window, and threshold. This is not a collision-risk clearance."
        return {
            "run_id": summary.get("run_id"),
            "window_start_utc": summary.get("window_start_utc"),
            "window_end_utc": summary.get("window_end_utc"),
            "threshold_km": _number(summary.get("config", {}).get("threshold_km")),
            "screened": screened, "shared_orbit_pairs_excluded": shared_pairs,
            "total": len(selected), "events": events, "note": note,
        }


def build_object_info(service, norad_id, root):
    """Return a JSON-safe current object profile; unknown IDs raise KeyError."""
    now = utc_now()
    with service.catalog_lock:
        index = service.index.get(norad_id)
        if index is None:
            raise KeyError(f"Unknown catalog object {norad_id}")
        row = deepcopy(service.rows[index])
        errors, positions, velocities = propagate([service.satellites[index]], now, [0.0])
        source_manifest = deepcopy(service.manifest)
    position = [_number(v) for v in positions[0, 0]]
    velocity = [_number(v) for v in velocities[0, 0]]
    valid = int(errors[0, 0]) == 0 and None not in position + velocity
    if not valid:
        position = velocity = None
    state = {
        "time_utc": utc_text(now), "frame": "TEME", "position_km": position,
        "velocity_km_s": velocity,
        "altitude_km": float(np.linalg.norm(position) - EARTH_RADIUS_KM) if valid else None,
        "speed_km_s": float(np.linalg.norm(velocity)) if valid else None,
        "sgp4_error": int(errors[0, 0]),
        "note": "SGP4 model estimate. Altitude is geocentric radius minus 6378.137 km, not geodetic height or a live measurement.",
    }
    epoch = _utc(row.get("EPOCH"))
    mean_motion = _number(row.get("MEAN_MOTION"))
    facts = []
    for label, key, unit in (("Inclination", "INCLINATION", "deg"),
                             ("Eccentricity", "ECCENTRICITY", None),
                             ("Mean motion", "MEAN_MOTION", "rev/day"),
                             ("Ascending node", "RA_OF_ASC_NODE", "deg"),
                             ("Argument of pericenter", "ARG_OF_PERICENTER", "deg"),
                             ("Mean anomaly", "MEAN_ANOMALY", "deg")):
        fact = {"label": label, "value": _number(row.get(key))}
        if unit:
            fact["unit"] = unit
        facts.append(fact)
    facts.append({"label": "Mean-element orbital period", "value": 1440 / mean_motion if mean_motion and mean_motion > 0 else None, "unit": "min"})
    profile = _history(row)
    histories, skipped = _snapshot_index(_snapshot_signature(root))
    history = deepcopy(histories.get(norad_id, []))
    sources = [deepcopy(_knowledge()["sources"][key])
               for key in dict.fromkeys(["celestrak-gp", *profile.get("source_ids", [])])]
    for item in source_manifest.get("sources", []):
        group = item.get("group")
        if group in row.get("SOURCE_GROUPS", []) and isinstance(group, str) and re.fullmatch(r"[a-z0-9-]+", group):
            sources.append({"id": f"celestrak-{group}", "title": f"CelesTrak — {group} GP catalog",
                            "url": f"https://celestrak.org/NORAD/elements/gp.php?GROUP={group}&FORMAT=JSON"})
    history_note = "Up to 8 distinct element epochs from the latest 32 locally saved CelesTrak snapshots, newest first. These are catalog element updates, not measured position tracks or a complete orbital history."
    if skipped:
        history_note += f" {skipped} unavailable or invalid snapshot(s) were skipped."
    if not history:
        history_note += " No saved snapshots contain this object."
    return {
        "id": norad_id, "name": str(row.get("OBJECT_NAME") or "Unnamed object"),
        "international_designator": row.get("OBJECT_ID") or None,
        "category": profile["category"], "groups": row.get("SOURCE_GROUPS", []),
        "epoch_utc": epoch,
        "element_age_hours": (now - parse_epoch(epoch)).total_seconds() / 3600 if epoch else None,
        "facts": facts, "history_summary": profile["history_summary"],
        "history_scope": profile["history_scope"], "history_events": profile["history_events"],
        "sources": sources, "catalog_history": history, "catalog_history_note": history_note,
        "current_state": state, "encounters": _encounters(service, norad_id),
    }
