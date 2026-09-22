"""Publish verified three-model artifacts to the local dashboard's static assets.

Only completed, replay-verified scientific run files are copied. This deliberately
does not expose the project directory or any LLM settings. Original runs remain
unchanged, including their historical publication metadata.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
from urllib.parse import quote
import uuid
import zipfile

ROOT = Path(__file__).resolve().parent
RUN_PATTERN = re.compile(r"\d{8}T\d{6}_\d{6}Z")
TOP_FILES = {"comparison_report.md", "config.json", "summary.json", "hardware.json", "status.json", "run.log"}
DIRECTORIES = {"sgp4", "pinn", "combined", "comparison", "reference", "inputs", "source_code"}
EXTENSIONS = {".json", ".jsonl", ".csv", ".npz", ".png", ".svg", ".md", ".pt", ".all", ".eof", ".txt", ".zip", ".py", ".ps1", ".bat"}
PAYLOAD_FILES = {"status.json", "summary.json", "pinn/training_history.csv", "combined/training_history.csv",
                 "comparison/physics_diagnostics.json", "comparison/verification.json",
                 "comparison/unavailable_collision_metrics.json"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe_source(folder, relative):
    path = (folder/relative).resolve()
    if not path.is_relative_to(folder) or not path.is_file() or path.stat().st_size > 100_000_000:
        raise ValueError(f"Invalid scientific source file: {relative}")
    return path


def atomic_json(path, data):
    path = Path(path)
    temporary = path.with_name(path.name+"."+uuid.uuid4().hex+".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    os.replace(temporary, path)


def allowed_file(relative):
    pure = PurePosixPath(relative)
    if pure.is_absolute() or "\\" in relative or ":" in relative or any(p in {".", ".."} or p.startswith(".") for p in pure.parts):
        return False
    return (relative in TOP_FILES or
            len(pure.parts) > 1 and pure.parts[0] in DIRECTORIES and pure.suffix.lower() in EXTENSIONS)


def history(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key, value in row.items():
            if value == "": row[key] = None
            else:
                try: row[key] = float(value)
                except ValueError: pass
    return rows


def publish_run(run_folder, public_root=None):
    run_folder = Path(run_folder).resolve()
    public_root = Path(public_root or ROOT/"dashboard/model-results").resolve()
    run_id = run_folder.name
    if not RUN_PATTERN.fullmatch(run_id):
        raise ValueError("Invalid model run ID")
    if read_json(safe_source(run_folder, "status.json")).get("status") != "complete":
        raise ValueError("Only completed experiments can be published")
    replay = read_json(safe_source(run_folder, "comparison/replay_verification.json"))
    if replay.get("passed") is not True:
        raise ValueError("The saved model replay verification must pass first")
    checksums = read_json(safe_source(run_folder, "file_checksums.json"))
    if not isinstance(checksums, dict) or not checksums:
        raise ValueError("Run checksums are missing")
    if not PAYLOAD_FILES.issubset(checksums):
        raise ValueError("Every dashboard payload source must be covered by the run checksum manifest")
    files, size = [], 0
    for relative, expected in checksums.items():
        if not allowed_file(relative):
            raise ValueError(f"Unexpected file in scientific run: {relative}")
        source = (run_folder/relative).resolve()
        if not source.is_relative_to(run_folder) or not source.is_file() or sha(source) != expected:
            raise ValueError(f"Scientific artifact checksum mismatch: {relative}")
        file_size = source.stat().st_size
        size += file_size
        if file_size > 100_000_000 or size > 500_000_000:
            raise ValueError("Scientific run exceeds the dashboard publication size limit")
        files.append(relative)
    # These two audit files are intentionally written after the original manifest.
    files += ["file_checksums.json", "comparison/replay_verification.json"]
    files = sorted(set(files))
    summary = read_json(run_folder/"summary.json")
    if summary.get("run_id") != run_id:
        raise ValueError("Run summary identity does not match its folder")
    if any(relative not in files for relative in summary.get("figures", [])):
        raise ValueError("Every declared figure must be a verified run artifact")
    for name in ("SGP4", "PINN", "COMBINED"):
        if not all(split in summary["metrics"][name] for split in ("train", "validation", "test")):
            raise ValueError("All three models and all evaluation splits are required")
    base_url = "/assets/model-results/"+run_id+"/"
    file_url = lambda relative:base_url+"/".join(quote(p, safe="") for p in PurePosixPath(relative).parts)
    entry = {k:summary[k] for k in ("run_id", "object_name", "norad_id", "start_utc", "end_utc")}
    entry.update(payload_url=base_url+"payload.json", base_url=base_url)
    public_root.mkdir(parents=True, exist_ok=True)
    destination = (public_root/run_id).resolve()
    if not destination.is_relative_to(public_root):
        raise ValueError("Publication destination escaped the asset folder")
    source_hashes = {relative:sha(run_folder/relative) for relative in files}
    if destination.exists():
        published = read_json(destination/"publication.json")
        if published.get("source_hashes") != source_hashes:
            raise ValueError("Published run changed; preserve it and generate a new experiment run")
        for relative, expected in published["published_hashes"].items():
            path = (destination/relative).resolve()
            if not path.is_relative_to(destination) or sha(path) != expected:
                raise ValueError(f"Published artifact changed: {relative}")
        return entry
    # Staging is outside the served dashboard. Both final paths are checked before the move.
    stage_root = (public_root.parent.parent/".local/model-publish").resolve()
    stage_root.mkdir(parents=True, exist_ok=True)
    stage = (stage_root/uuid.uuid4().hex).resolve()
    if not stage.is_relative_to(stage_root):
        raise ValueError("Invalid publication staging path")
    stage.mkdir()
    payload = {k:summary[k] for k in ("run_id", "object_name", "norad_id", "start_utc", "end_utc", "reference_label",
               "metrics", "training_proofs", "hardware", "prediction_timing_seconds", "split", "figures")}
    payload.update(base_url=base_url, archive_url=base_url+"complete_run.zip", summary_url=base_url+"summary.json",
                   website_publication=True, retrospective=True, reference_is_final_precise=False,
                   timing_note=summary.get("timing_note", ""),
                   histories={"PINN":history(run_folder/"pinn/training_history.csv"),
                              "COMBINED":history(run_folder/"combined/training_history.csv")},
                   physics_diagnostics=read_json(run_folder/"comparison/physics_diagnostics.json"),
                   unavailable_collision_metrics=read_json(run_folder/"comparison/unavailable_collision_metrics.json"),
                   verification=read_json(run_folder/"comparison/verification.json"), replay_verification=replay,
                   files=[{"path":relative, "bytes":(run_folder/relative).stat().st_size, "url":file_url(relative)} for relative in files])
    for relative in files:
        target = stage/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(run_folder/relative, target)
        if sha(target) != source_hashes[relative]:
            raise ValueError("Run changed during publication")
    atomic_json(stage/"payload.json", payload)
    with zipfile.ZipFile(stage/"complete_run.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative in files:
            archive.write(stage/relative, run_id+"/"+relative)
    published_hashes = {relative:sha(stage/relative) for relative in files+["payload.json", "complete_run.zip"]}
    atomic_json(stage/"publication.json", {"run_id":run_id, "published_at_utc":datetime.now(timezone.utc).isoformat(),
                "source_hashes":source_hashes, "published_hashes":published_hashes, "original_results_unchanged":True,
                "note":"Published to the local dashboard at the user's request. Original reports retain their creation-time publication statements."})
    if not stage.resolve().is_relative_to(stage_root) or not destination.resolve().is_relative_to(public_root):
        raise ValueError("Publication path check failed")
    os.replace(stage, destination)
    return entry


def publish_all(results_root=None, public_root=None):
    results_root = Path(results_root or ROOT/"model_results").resolve()
    public_root = Path(public_root or ROOT/"dashboard/model-results").resolve()
    entries = []
    for folder in sorted(results_root.iterdir(), reverse=True) if results_root.exists() else []:
        if folder.is_dir() and RUN_PATTERN.fullmatch(folder.name) and (folder/"status.json").exists():
            if read_json(folder/"status.json").get("status") == "complete":
                entries.append(publish_run(folder, public_root))
    public_root.mkdir(parents=True, exist_ok=True)
    index = {"schema_version":1, "latest_run_id":entries[0]["run_id"] if entries else None, "runs":entries}
    atomic_json(public_root/"index.json", index)
    print(f"Published {len(entries)} verified model experiment(s) to the local dashboard.", flush=True)
    return index


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--public-root", type=Path)
    args = parser.parse_args()
    publish_all(args.results_root, args.public_root)
