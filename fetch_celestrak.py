"""Step 1: download and validate CelesTrak GP orbital elements (Python 3.10+)."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener


BASE_URL = "https://celestrak.org/NORAD/elements/gp.php"
DEFAULT_GROUPS = (
    "stations", "cosmos-2251-debris", "iridium-33-debris", "fengyun-1c-debris"
)
CACHE_LIFETIME = timedelta(hours=2)
NUMERIC_FIELDS = (
    "MEAN_MOTION", "ECCENTRICITY", "INCLINATION", "RA_OF_ASC_NODE",
    "ARG_OF_PERICENTER", "MEAN_ANOMALY", "BSTAR",
    "MEAN_MOTION_DOT", "MEAN_MOTION_DDOT",
)
CSV_FIRST_FIELDS = (
    "NORAD_CAT_ID", "OBJECT_NAME", "OBJECT_ID", "EPOCH", *NUMERIC_FIELDS,
    "EPHEMERIS_TYPE", "CLASSIFICATION_TYPE", "ELEMENT_SET_NO", "REV_AT_EPOCH",
    "SOURCE_GROUPS",
)


class DataError(Exception):
    """An actionable download, validation, or cache error."""


class NoRedirects(HTTPRedirectHandler):
    """CelesTrak requires stopping on non-200 responses, including redirects."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def utc_now():
    return datetime.now(timezone.utc)


def utc_text(value):
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_epoch(value):
    # CelesTrak omits the timezone suffix; its documented time system is UTC.
    if not isinstance(value, str) or "T" not in value:
        raise ValueError("expected an ISO-8601 timestamp")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def group_name(value):
    value = value.lower().strip()
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", value):
        raise argparse.ArgumentTypeError("use a CelesTrak group name, e.g. stations")
    return value


def positive_seconds(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("timeout must be a positive finite number")
    return number


def validate_records(payload):
    """Reject empty/error responses and unusable orbital data without changing it."""
    if not isinstance(payload, list) or not payload:
        raise DataError("Expected a non-empty JSON array of orbital elements.")
    for index, row in enumerate(payload):
        try:
            if not isinstance(row, dict):
                raise ValueError("record is not an object")
            catalog_id = row["NORAD_CAT_ID"]
            if isinstance(catalog_id, bool) or not re.fullmatch(r"[0-9]{1,9}", str(catalog_id)):
                raise ValueError("NORAD_CAT_ID must be a 1-9 digit integer")
            if int(catalog_id) < 1:
                raise ValueError("NORAD_CAT_ID must be positive")
            parse_epoch(row["EPOCH"])
            values = {}
            for field in NUMERIC_FIELDS:
                value = row[field]
                if isinstance(value, bool) or not math.isfinite(float(value)):
                    raise ValueError(f"{field} must be finite")
                values[field] = float(value)
            if values["MEAN_MOTION"] <= 0:
                raise ValueError("MEAN_MOTION must be positive")
            if not 0 <= values["ECCENTRICITY"] < 1:
                raise ValueError("ECCENTRICITY must be in [0, 1)")
            if not 0 <= values["INCLINATION"] <= 180:
                raise ValueError("INCLINATION must be in [0, 180] degrees")
            for field in ("RA_OF_ASC_NODE", "ARG_OF_PERICENTER", "MEAN_ANOMALY"):
                if not 0 <= values[field] < 360:
                    raise ValueError(f"{field} must be in [0, 360) degrees")
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise DataError(f"Invalid orbital record at index {index}: {exc}") from exc
    return payload


def write_json(path, payload):
    """Replace a complete file atomically; a failed write leaves the old file intact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def download_records(url, timeout):
    request = Request(url, headers={
        "User-Agent": "PINN-Space-Debris-Research/0.1",
        "Accept": "application/json",
    })
    try:
        with build_opener(NoRedirects()).open(request, timeout=timeout) as response:
            if response.status != 200:
                raise DataError(f"CelesTrak returned HTTP {response.status} for {url}.")
            body = response.read().decode("utf-8-sig")
    except HTTPError as exc:
        code = exc.code
        retry_after = exc.headers.get("Retry-After") if exc.headers else None
        exc.close()
        hint = f" Retry-After: {retry_after}." if retry_after else ""
        raise DataError(
            f"CelesTrak returned HTTP {code} for {url}.{hint} "
            "Stopped without retrying. Check the URL and CelesTrak usage policy; "
            "for 403/429, wait before trying again. Existing data is preserved."
        ) from exc
    except (URLError, OSError, UnicodeError) as exc:
        raise DataError(f"Could not download {url}: {exc}. No automatic retry.") from exc
    try:
        return validate_records(json.loads(body))
    except json.JSONDecodeError as exc:
        raise DataError(f"CelesTrak did not return valid JSON for {url}.") from exc


def fetch_group(group, output_dir, timeout=30, offline=False):
    group = group_name(group)
    url = BASE_URL + "?" + urlencode({"GROUP": group, "FORMAT": "JSON"})
    cache_file = output_dir / "cache" / f"{group}.json"
    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            if cached["group"] != group or cached["url"] != url:
                raise ValueError("cache source does not match the requested group")
            validate_records(cached["records"])
            age = utc_now() - parse_epoch(cached["fetched_at_utc"])
            if age < timedelta(0):
                raise ValueError("cache timestamp is in the future; check the system clock")
        except (KeyError, TypeError, ValueError, DataError) as exc:
            raise DataError(f"Invalid cache {cache_file}: {exc}. Inspect it before fetching again.") from exc
        if offline or age < CACHE_LIFETIME:
            return {**cached, "from_cache": True}
    if offline:
        raise DataError(f"No cached data for {group}. Run once with internet access first.")
    records = download_records(url, timeout)
    result = {
        "group": group, "url": url, "fetched_at_utc": utc_text(utc_now()),
        "records": records,
    }
    write_json(cache_file, result)
    return {**result, "from_cache": False}


def merge_records(datasets):
    """One row per catalog ID; retain the newest epoch and all group memberships."""
    merged = {}
    memberships = {}
    for dataset in datasets:
        for row in dataset["records"]:
            catalog_id = int(row["NORAD_CAT_ID"])
            memberships.setdefault(catalog_id, set()).add(dataset["group"])
            if (catalog_id not in merged or
                    parse_epoch(row["EPOCH"]) > parse_epoch(merged[catalog_id]["EPOCH"])):
                merged[catalog_id] = dict(row)
    return [
        {**merged[key], "NORAD_CAT_ID": key, "SOURCE_GROUPS": sorted(memberships[key])}
        for key in sorted(merged)
    ]


def save_snapshot(datasets, output_dir, offline=False):
    generated_at = utc_now()
    records = merge_records(datasets)
    snapshot = output_dir / "snapshots" / generated_at.strftime("%Y%m%dT%H%M%S_%fZ")
    snapshot.mkdir(parents=True, exist_ok=False)
    for dataset in datasets:
        write_json(snapshot / "raw" / f"{dataset['group']}.json", dataset["records"])
    write_json(snapshot / "orbital_data.json", records)
    extra_fields = sorted(set().union(*(row.keys() for row in records)) - set(CSV_FIRST_FIELDS))
    with (snapshot / "orbital_data.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*CSV_FIRST_FIELDS, *extra_fields])
        writer.writeheader()
        for row in records:
            writer.writerow({**row, "SOURCE_GROUPS": ";".join(row["SOURCE_GROUPS"])})
    epochs = [parse_epoch(row["EPOCH"]) for row in records]
    manifest = {
        "schema_version": 1,
        "generated_at_utc": utc_text(generated_at),
        "offline": offline,
        "unique_objects": len(records),
        "oldest_epoch_utc": utc_text(min(epochs)),
        "newest_epoch_utc": utc_text(max(epochs)),
        "objects_with_epoch_older_than_72_hours": sum(
            generated_at - epoch > timedelta(hours=72) for epoch in epochs
        ),
        "gp_context": {
            "CENTER_NAME": "EARTH", "REF_FRAME": "TEME", "TIME_SYSTEM": "UTC",
            "MEAN_ELEMENT_THEORY": "SGP4",
        },
        "sources": [
            {**{key: value for key, value in dataset.items() if key != "records"},
             "record_count": len(dataset["records"])}
            for dataset in datasets
        ],
    }
    write_json(snapshot / "manifest.json", manifest)
    # Publish the pointer only after every file in the complete snapshot is written.
    write_json(output_dir / "latest.json", {
        "snapshot": snapshot.relative_to(output_dir).as_posix(), **manifest,
    })
    return snapshot, manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", nargs="+", type=group_name, default=DEFAULT_GROUPS,
                        help="CelesTrak group names (default: stations and three debris groups)")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--timeout", type=positive_seconds, default=30,
                        help="HTTP timeout in seconds (default: 30)")
    parser.add_argument("--offline", action="store_true",
                        help="use cached data only, even if older than two hours")
    args = parser.parse_args(argv)
    try:
        datasets = []
        for group in dict.fromkeys(args.groups):
            print(f"Loading {group} ...", flush=True)
            dataset = fetch_group(group, args.output_dir, args.timeout, args.offline)
            datasets.append(dataset)
            source = "cache" if dataset["from_cache"] else "CelesTrak"
            print(f"  {len(dataset['records'])} objects from {source}; "
                  f"downloaded at {dataset['fetched_at_utc']}", flush=True)
        snapshot, manifest = save_snapshot(datasets, args.output_dir, args.offline)
    except (DataError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Saved {manifest['unique_objects']} unique objects to {snapshot.resolve()}")
    print(f"Epoch range: {manifest['oldest_epoch_utc']} to {manifest['newest_epoch_utc']}")
    print(f"Objects with epochs older than 72 hours: "
          f"{manifest['objects_with_epoch_older_than_72_hours']}")
    if args.offline:
        print("Offline run: cached download dates above may be old.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
