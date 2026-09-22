"""Local dashboard: live SGP4 states, background screening, and result downloads."""
from __future__ import annotations

import argparse
from datetime import timedelta
import json
from pathlib import Path
import re
import shutil
import threading
import time

import numpy as np
from flask import Flask, abort, jsonify, request, send_from_directory

from fetch_celestrak import DEFAULT_GROUPS, fetch_group, parse_epoch, save_snapshot, utc_now, utc_text
from orbit_engine import AnalysisConfig, prepare_catalog, propagate
from run_analysis import ROOT, read_snapshot, run_analysis
from chat_service import ChatError, ChatService, validate_conversation
from object_context import build_object_info


class DashboardService:
    def __init__(self):
        self.lock = threading.RLock()
        self.catalog_lock = threading.RLock()
        self.archive_lock = threading.Lock()
        self.job = {"running": False, "progress": 0, "message": "Ready", "error": None}
        self.auto_refresh = True
        self.last_attempt = time.monotonic()
        self.summary = None
        self.events = []
        self.config = AnalysisConfig()
        self.reload_catalog()
        self.reload_results()

    def reload_catalog(self):
        _, manifest, records = read_snapshot()
        satellites, rows, excluded = prepare_catalog(records)
        with self.catalog_lock:
            self.manifest, self.satellites, self.rows = manifest, satellites, rows
            self.catalog_errors = excluded
            self.index = {int(row["NORAD_CAT_ID"]): i for i, row in enumerate(rows)}

    def reload_results(self):
        pointer = ROOT / "results" / "latest.json"
        if pointer.exists():
            run_id = json.loads(pointer.read_text(encoding="utf-8"))["run_id"]
            folder = run_folder(run_id)
            summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
            events = json.loads((folder / "conjunctions.json").read_text(encoding="utf-8"))
            with self.lock:
                self.summary, self.events = summary, events
                self.config = AnalysisConfig(**summary["config"])

    def launch(self, config, refresh=False):
        config.validate()
        with self.lock:
            if self.job["running"]:
                raise RuntimeError("An analysis or refresh is already running")
            self.job = {"running": True, "progress": 0, "message": "Starting", "error": None}
            self.config = config
        def progress(percent, message):
            with self.lock:
                self.job.update(progress=percent, message=message)
        def work():
            try:
                if refresh:
                    progress(0, "Checking CelesTrak data (two-hour cache applies)")
                    self.last_attempt = time.monotonic()
                    groups = [source["group"] for source in self.manifest["sources"]] or list(DEFAULT_GROUPS)
                    datasets = [fetch_group(group, ROOT / "data") for group in groups]
                    save_snapshot(datasets, ROOT / "data")
                    self.reload_catalog()
                run_analysis(config=config, progress=progress)
                self.reload_results()
            except Exception as exc:
                with self.lock:
                    self.job.update(error=str(exc), message="Run failed; previous results preserved")
                    if refresh:
                        self.auto_refresh = False
            finally:
                with self.lock:
                    self.job["running"] = False
        threading.Thread(target=work, daemon=True, name="sgp4-analysis").start()

    def scheduler(self):
        while True:
            time.sleep(30)
            with self.lock:
                if not self.auto_refresh or self.job["running"]:
                    continue
                fetch_times = [parse_epoch(source["fetched_at_utc"]) for source in self.manifest["sources"]]
                due = (utc_now() - min(fetch_times)) >= timedelta(hours=2)
                if due and time.monotonic() - self.last_attempt >= 120:
                    try:
                        self.launch(self.config, refresh=True)
                    except RuntimeError:
                        pass

    def live(self):
        now = utc_now()
        with self.catalog_lock:
            errors, positions, velocities = propagate(self.satellites, now, [0.0])
            states = []
            for index, row in enumerate(self.rows):
                vector = [*positions[index, 0], *velocities[index, 0]]
                states.append([int(row["NORAD_CAT_ID"]),
                               *[round(float(value), 6) if np.isfinite(value) else None for value in vector],
                               int(errors[index, 0])])
        return {"time_utc": utc_text(now), "frame": "TEME", "states": states,
                "columns": ["norad_id", "x_km", "y_km", "z_km", "vx_km_s", "vy_km_s", "vz_km_s", "sgp4_error"]}


def run_folder(run_id):
    if not re.fullmatch(r"\d{8}T\d{6}_\d{6}Z", run_id):
        raise ValueError("Invalid run ID")
    folder = ROOT / "results" / run_id
    if not (folder / "status.json").exists():
        raise ValueError("Unknown run")
    if json.loads((folder / "status.json").read_text(encoding="utf-8"))["status"] != "complete":
        raise ValueError("Run is not complete")
    return folder


def create_app(service=None, chat_service=None):
    app = Flask(__name__, static_folder=str(ROOT / "dashboard"), static_url_path="/assets")
    app.config.update(MAX_CONTENT_LENGTH=32768, TRUSTED_HOSTS=["127.0.0.1", "localhost"])
    service = service or DashboardService()
    app.extensions["orbit_service"] = service
    chat_service = chat_service or ChatService(ROOT)
    app.extensions["chat_service"] = chat_service

    @app.before_request
    def local_mutations_only():
        if request.method == "POST":
            if not request.is_json:
                abort(415)
            origin = request.headers.get("Origin")
            if origin and origin != request.host_url.rstrip("/"):
                abort(403)

    @app.after_request
    def response_headers(response):
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'"
        return response

    @app.errorhandler(ValueError)
    @app.errorhandler(KeyError)
    @app.errorhandler(TypeError)
    def invalid_request(error):
        return jsonify(error=str(error)), 400

    @app.errorhandler(ChatError)
    def chat_error(error):
        return jsonify(error=str(error)), error.status_code

    @app.errorhandler(413)
    def oversized_request(error):
        return jsonify(error="The request is too large. Shorten it or clear the conversation."), 413

    @app.get("/api/objects/<int:norad_id>/info")
    def object_info(norad_id):
        try:
            return jsonify(build_object_info(service, norad_id, ROOT))
        except KeyError:
            return jsonify(error="This object is not in the current catalog."), 404

    @app.get("/api/chat/status")
    def chat_status():
        return jsonify(chat_service.status())

    @app.post("/api/chat/config")
    def chat_config():
        return jsonify(chat_service.configure(request.get_json()))

    @app.post("/api/chat")
    def chat():
        payload = request.get_json()
        if not isinstance(payload, dict) or set(payload) - {"object_id", "message", "history", "provider"}:
            raise ValueError("Send object_id, message, and optional history and provider.")
        if "provider" not in payload:
            raise ChatError("Refresh this page to load the Gemini chat before sending again.", 409)
        if payload["provider"] != "gemini":
            raise ValueError("This dashboard supports only the Gemini API.")
        norad_id = payload.get("object_id")
        if type(norad_id) is not int or not 1 <= norad_id <= 999999999:
            raise ValueError("Select a valid NORAD catalog object.")
        message, history = validate_conversation(payload.get("message"), payload.get("history", []))
        try:
            context = build_object_info(service, norad_id, ROOT)
        except KeyError:
            return jsonify(error="This object is not in the current catalog."), 404
        return jsonify(chat_service.answer(context, message, history, expected_provider="gemini"))

    @app.get("/")
    def home():
        return send_from_directory(ROOT / "dashboard", "index.html")

    @app.get("/api/state")
    def state():
        with service.lock:
            return jsonify(job=service.job, summary=service.summary, data=service.manifest,
                           auto_refresh=service.auto_refresh, config=service.config.to_dict(),
                           server_time_utc=utc_text(utc_now()))

    @app.get("/api/catalog")
    def catalog():
        with service.catalog_lock:
            return jsonify(objects=[{"id": int(row["NORAD_CAT_ID"]), "name": row.get("OBJECT_NAME", "Unnamed object"),
                                     "epoch": row["EPOCH"], "groups": row.get("SOURCE_GROUPS", []),
                                     "mean_motion": row["MEAN_MOTION"]} for row in service.rows],
                           generated_at_utc=service.manifest["generated_at_utc"])

    @app.get("/api/live")
    def live():
        return jsonify(service.live())

    @app.get("/api/orbit/<int:norad_id>")
    def orbit(norad_id):
        now = utc_now()
        with service.catalog_lock:
            index = service.index.get(norad_id)
            if index is None:
                abort(404)
            duration = min(172800.0, 86400 / float(service.rows[index]["MEAN_MOTION"]))
            seconds = np.linspace(0, duration, 181)
            errors, positions, _ = propagate([service.satellites[index]], now, seconds)
        return jsonify(time_utc=utc_text(now), duration_seconds=duration,
                       positions=[[float(value) for value in p] if code == 0 and np.isfinite(p).all() else None
                                  for p, code in zip(positions[0], errors[0])])

    @app.get("/api/conjunctions")
    def conjunctions():
        with service.lock:
            return jsonify(run_id=service.summary["run_id"] if service.summary else None, events=service.events)

    @app.post("/api/analyze")
    def analyze():
        config = AnalysisConfig(**request.get_json()).validate()
        try:
            service.launch(config)
        except RuntimeError as exc:
            return jsonify(error=str(exc)), 409
        return jsonify(status="started"), 202

    @app.post("/api/refresh")
    def refresh():
        try:
            service.launch(service.config, refresh=True)
        except RuntimeError as exc:
            return jsonify(error=str(exc)), 409
        return jsonify(status="started"), 202

    @app.post("/api/auto-refresh")
    def auto_refresh():
        enabled = request.get_json()["enabled"]
        if not isinstance(enabled, bool):
            raise ValueError("enabled must be true or false")
        with service.lock:
            service.auto_refresh = enabled
            if enabled:
                service.last_attempt = time.monotonic()
        return jsonify(enabled=enabled)

    @app.get("/api/download/<run_id>/<filename>")
    def download(run_id, filename):
        folder = run_folder(run_id)
        allowed = {"report.md", "summary.json", "metrics.json", "metrics.csv", "conjunctions.csv",
                   "conjunctions.json", "trajectories.npz", "objects_and_initial_states.csv", "labels_template.json"}
        if filename == "all.zip":
            with service.archive_lock:
                target = ROOT / "results" / "downloads" / run_id
                target.parent.mkdir(exist_ok=True)
                if not target.with_suffix(".zip").exists():
                    shutil.make_archive(str(target), "zip", root_dir=folder.parent, base_dir=folder.name)
            return send_from_directory(target.parent, target.name + ".zip", as_attachment=True)
        if filename not in allowed:
            abort(404)
        return send_from_directory(folder, filename, as_attachment=True)

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--no-auto-refresh", action="store_true")
    args = parser.parse_args()
    service = DashboardService()
    service.auto_refresh = not args.no_auto_refresh
    threading.Thread(target=service.scheduler, daemon=True, name="celestrak-refresh").start()
    print(f"Dashboard: http://127.0.0.1:{args.port}", flush=True)
    create_app(service).run(host="127.0.0.1", port=args.port, threaded=True, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
