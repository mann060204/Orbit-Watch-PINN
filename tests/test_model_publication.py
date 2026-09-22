"""Small synthetic publication fixtures; these are not orbit or training results.

Every test uses a temporary tree. No saved scientific run or real credential is
opened, modified, or copied by this test module.
"""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from publish_model_results import allowed_file, publish_all, publish_run


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ModelPublicationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.results = self.root / "model_results"
        self.public = self.root / "dashboard" / "model-results"

    def fixture(self, run_id="20260101T000000_000001Z"):
        """Create identifiable unit-test bytes, not a pretend trained model."""
        run = self.results / run_id
        summary = {
            "run_id": run_id, "object_name": "UNIT TEST FIXTURE", "norad_id": 1,
            "start_utc": "2026-01-01T00:00:00Z", "end_utc": "2026-01-01T01:00:00Z",
            "reference_label": "Synthetic publication unit test; not an orbit",
            "metrics": {model: {split: {"position_vector_rmse_km": index / 1000}
                         for index, split in enumerate(("train", "validation", "test"), 1)}
                        for model in ("SGP4", "PINN", "COMBINED")},
            "training_proofs": {"fixture_only": True}, "hardware": {"selected_device": "test"},
            "prediction_timing_seconds": {}, "split": {"website_publication": False},
            "figures": ["sgp4/figures/test.svg"], "website_publication": False,
        }
        documents = {
            "summary.json": summary,
            "status.json": {"status": "complete"},
            "comparison/physics_diagnostics.json": {"fixture_only": True},
            "comparison/verification.json": {"fixture_only": True, "passed": True},
            "comparison/unavailable_collision_metrics.json": {"collision_probability": None},
        }
        for relative, value in documents.items():
            write_json(run / relative, value)
        for model in ("pinn", "combined"):
            folder = run / model
            folder.mkdir(parents=True)
            (folder / "model.pt").write_bytes(b"UNIT TEST BYTES; NOT A TRAINED MODEL\x00\xff" + model.encode())
            (folder / "training_history.csv").write_text(
                "step,loss,label,optional\n1,0.125,unit test,\n2,0.03125,unit test,\n", encoding="utf-8")
        figures = run / "sgp4" / "figures"
        figures.mkdir(parents=True)
        (figures / "test.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg"><title>Unit test</title></svg>', encoding="utf-8")
        (run / "comparison_report.md").write_text("Unit test fixture only.\n", encoding="utf-8")
        self.sign_fixture(run)
        # The actual replay audit is generated after the scientific manifest.
        write_json(run / "comparison/replay_verification.json", {"passed": True, "fixture_only": True})
        return run, summary

    def sign_fixture(self, run):
        manifest = {str(p.relative_to(run)).replace("\\", "/"): digest(p)
                    for p in run.rglob("*") if p.is_file()
                    and p.relative_to(run).as_posix() not in {"file_checksums.json", "comparison/replay_verification.json"}}
        write_json(run / "file_checksums.json", manifest)
        return manifest

    def test_published_metrics_and_model_bytes_equal_source(self):
        run, summary = self.fixture()
        before = {p.relative_to(run).as_posix(): p.read_bytes() for p in run.rglob("*") if p.is_file()}
        entry = publish_run(run, self.public)
        destination = self.public / run.name
        payload = json.loads((destination / "payload.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["metrics"], summary["metrics"])
        self.assertTrue(payload["website_publication"])
        self.assertTrue(payload["replay_verification"]["passed"])
        self.assertEqual(payload["histories"]["PINN"][0]["loss"], 0.125)
        self.assertIsNone(payload["histories"]["PINN"][0]["optional"])
        self.assertEqual(entry["run_id"], run.name)
        self.assertEqual(entry["payload_url"], "/assets/model-results/" + run.name + "/payload.json")
        for relative, contents in before.items():
            with self.subTest(file=relative):
                self.assertEqual((destination / relative).read_bytes(), contents)
                self.assertEqual((run / relative).read_bytes(), contents, "Publication must preserve original runs")

    def test_archive_and_file_index_exclude_unsigned_private_files(self):
        run, _ = self.fixture()
        private_paths = [".local/chat_config.json", ".env", "inputs/credentials.json", "pinn/unlisted.txt"]
        for relative in private_paths:
            target = run / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("UNIT TEST CREDENTIAL MARKER", encoding="utf-8")
        publish_run(run, self.public)
        destination = self.public / run.name
        payload = json.loads((destination / "payload.json").read_text(encoding="utf-8"))
        published = {item["path"] for item in payload["files"]}
        with zipfile.ZipFile(destination / "complete_run.zip") as archive:
            self.assertEqual(set(archive.namelist()), {run.name + "/" + relative for relative in published})
            for name in archive.namelist():
                self.assertNotIn(b"UNIT TEST CREDENTIAL MARKER", archive.read(name))
        for relative in private_paths:
            self.assertNotIn(relative, published)
            self.assertFalse((destination / relative).exists())

    def test_incomplete_or_failed_runs_are_rejected(self):
        run, _ = self.fixture()
        for state in ("running", "failed", "cancelled"):
            with self.subTest(status=state):
                write_json(run / "status.json", {"status": state})
                with self.assertRaisesRegex(ValueError, "completed"):
                    publish_run(run, self.public)
                self.assertFalse((self.public / run.name).exists())

    def test_replay_must_explicitly_pass(self):
        run, _ = self.fixture()
        for passed in (False, None, "true", 1):
            with self.subTest(passed=passed):
                write_json(run / "comparison/replay_verification.json", {"passed": passed})
                with self.assertRaisesRegex(ValueError, "replay"):
                    publish_run(run, self.public)
                self.assertFalse((self.public / run.name).exists())

    def test_checksum_failure_never_publishes_run(self):
        run, _ = self.fixture()
        (run / "pinn/model.pt").write_bytes(b"changed unit test bytes")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            publish_run(run, self.public)
        self.assertFalse((self.public / run.name).exists())

    def test_missing_signed_artifact_is_rejected(self):
        run, _ = self.fixture()
        (run / "combined/model.pt").unlink()
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            publish_run(run, self.public)
        self.assertFalse((self.public / run.name).exists())

    def test_all_payload_inputs_require_a_checksum(self):
        run, _ = self.fixture()
        manifest = json.loads((run / "file_checksums.json").read_text(encoding="utf-8"))
        for relative in ("summary.json", "status.json", "pinn/training_history.csv", "combined/training_history.csv",
                         "comparison/physics_diagnostics.json", "comparison/verification.json",
                         "comparison/unavailable_collision_metrics.json"):
            with self.subTest(missing_checksum=relative):
                write_json(run / "file_checksums.json", {key: value for key, value in manifest.items() if key != relative})
                with self.assertRaises(ValueError):
                    publish_run(run, self.public)
                self.assertFalse((self.public / run.name).exists())

    def test_private_and_traversal_paths_in_manifest_are_rejected(self):
        run, _ = self.fixture()
        manifest = json.loads((run / "file_checksums.json").read_text(encoding="utf-8"))
        paths = [".local/chat_config.json", ".env", "inputs/.env.json", "../summary.json",
                 "sgp4/../../summary.json", "/summary.json", "C:/private/key.json", "sgp4\\metrics.json"]
        for relative in paths:
            with self.subTest(path=relative):
                self.assertFalse(allowed_file(relative))
                write_json(run / "file_checksums.json", {**manifest, relative: "0" * 64})
                with self.assertRaisesRegex(ValueError, "Unexpected file"):
                    publish_run(run, self.public)
                self.assertFalse((self.public / run.name).exists())

    def test_unknown_top_level_files_are_not_allowed(self):
        for relative in ("chat_config.json", "secrets.txt", "private/key.json", "inputs/token.exe"):
            with self.subTest(path=relative):
                self.assertFalse(allowed_file(relative))

    def test_repeat_publication_preserves_all_bytes_and_audit_timestamp(self):
        run, _ = self.fixture()
        first = publish_run(run, self.public)
        destination = self.public / run.name
        before = {p.relative_to(destination).as_posix(): p.read_bytes() for p in destination.rglob("*") if p.is_file()}
        second = publish_run(run, self.public)
        self.assertEqual(first, second)
        after = {p.relative_to(destination).as_posix(): p.read_bytes() for p in destination.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_changed_source_with_new_checksums_does_not_replace_published_run(self):
        run, _ = self.fixture()
        publish_run(run, self.public)
        destination_model = self.public / run.name / "combined/model.pt"
        original_model = destination_model.read_bytes()
        (run / "combined/model.pt").write_bytes(b"new unit test weights")
        self.sign_fixture(run)
        with self.assertRaisesRegex(ValueError, "Published run changed"):
            publish_run(run, self.public)
        self.assertEqual(destination_model.read_bytes(), original_model)

    def test_modified_published_artifact_is_reported_on_repeat(self):
        run, _ = self.fixture()
        publish_run(run, self.public)
        (self.public / run.name / "pinn/model.pt").write_bytes(b"modified published test weights")
        with self.assertRaisesRegex(ValueError, "Published artifact changed"):
            publish_run(run, self.public)

    def test_missing_evaluation_split_is_rejected(self):
        run, summary = self.fixture()
        del summary["metrics"]["COMBINED"]["test"]
        write_json(run / "summary.json", summary)
        self.sign_fixture(run)
        with self.assertRaisesRegex(ValueError, "all evaluation splits"):
            publish_run(run, self.public)

    def test_unsigned_or_external_declared_figures_are_rejected(self):
        run, summary = self.fixture()
        for figure in ("sgp4/figures/unsigned.svg", "https://example.com/plot.svg", "../private.svg"):
            with self.subTest(figure=figure):
                summary["figures"] = [figure]
                write_json(run / "summary.json", summary)
                self.sign_fixture(run)
                with self.assertRaisesRegex(ValueError, "declared figure"):
                    publish_run(run, self.public)
                self.assertFalse((self.public / run.name).exists())

    def test_summary_cannot_claim_another_run_id(self):
        run, summary = self.fixture()
        summary["run_id"] = "20260102T000000_000001Z"
        write_json(run / "summary.json", summary)
        self.sign_fixture(run)
        with self.assertRaisesRegex(ValueError, "summary identity"):
            publish_run(run, self.public)
        self.assertFalse((self.public / run.name).exists())

    def test_index_orders_by_run_id_and_skips_newer_failed_run(self):
        older, _ = self.fixture("20260101T000000_000001Z")
        newer, _ = self.fixture("20260102T000000_000001Z")
        failed = self.results / "20260103T000000_000001Z"
        write_json(failed / "status.json", {"status": "failed"})
        write_json(self.results / "latest.json", {"run_id": failed.name})
        index = publish_all(self.results, self.public)
        self.assertEqual(index["latest_run_id"], newer.name)
        self.assertEqual([entry["run_id"] for entry in index["runs"]], [newer.name, older.name])
        self.assertFalse((self.public / failed.name).exists())
        self.assertEqual(json.loads((self.public / "index.json").read_text(encoding="utf-8")), index)

    def test_empty_results_directory_produces_empty_index(self):
        self.results.mkdir()
        index = publish_all(self.results, self.public)
        self.assertIsNone(index["latest_run_id"])
        self.assertEqual(index["runs"], [])

    def test_missing_results_directory_produces_empty_index(self):
        self.assertFalse(self.results.exists())
        index = publish_all(self.results, self.public)
        self.assertIsNone(index["latest_run_id"])
        self.assertEqual(index["runs"], [])
        self.assertTrue((self.public / "index.json").is_file())


if __name__ == "__main__":
    unittest.main()
