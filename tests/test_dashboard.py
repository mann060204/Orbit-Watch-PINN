from types import SimpleNamespace
import threading
import unittest
from unittest.mock import Mock

from dashboard_server import create_app
from orbit_engine import AnalysisConfig


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.service=SimpleNamespace(lock=threading.RLock(), catalog_lock=threading.RLock(),
            summary=None, job={"running":False}, manifest={"sources":[],"generated_at_utc":"2026-09-10T00:00:00Z"},
            config=AnalysisConfig(),auto_refresh=True,events=[],rows=[],
            archive_lock=threading.Lock(),
            launch=Mock(),live=Mock(return_value={"time_utc":"2026-09-10T00:00:00Z","states":[]}))
        self.client=create_app(self.service).test_client()

    def test_page_assets_and_read_routes(self):
        for path in ["/","/assets/app.js","/assets/styles.css","/assets/assistant.js","/assets/assistant.css",
                     "/assets/model-results.js","/assets/model-results.css",
                     "/api/state","/api/catalog","/api/live","/api/conjunctions"]:
            with self.subTest(path=path):
                response=self.client.get(path)
                self.assertEqual(response.status_code,200)
                response.close()

    def test_valid_run_starts_background_job(self):
        response=self.client.post("/api/analyze",json=AnalysisConfig().to_dict())
        self.assertEqual(response.status_code,202)
        self.service.launch.assert_called_once()

    def test_invalid_run_and_foreign_origin_rejected(self):
        self.assertEqual(self.client.post("/api/analyze",json={"step_seconds":0}).status_code,400)
        self.assertEqual(self.client.post("/api/analyze",json={},headers={"Origin":"https://example.com"}).status_code,403)
        self.assertEqual(self.client.post("/api/analyze",data="x").status_code,415)
        self.service.launch.assert_not_called()

    def test_busy_run_returns_conflict(self):
        self.service.launch.side_effect=RuntimeError("Already running")
        self.assertEqual(self.client.post("/api/analyze",json={}).status_code,409)

    def test_auto_refresh_needs_boolean(self):
        self.assertEqual(self.client.post("/api/auto-refresh",json={"enabled":"false"}).status_code,400)
        self.assertEqual(self.client.post("/api/auto-refresh",json={"enabled":False}).status_code,200)
        self.assertFalse(self.service.auto_refresh)

    def test_old_training_endpoints_and_raw_private_directories_stay_unpublished(self):
        for path in ["/api/pinn", "/assets/pinn.js", "/assets/pinn.css",
                     "/api/pinn/download/20260911T000000_000000Z/model.pt",
                     "/pinn_results/latest.json", "/comparison_results/latest.json",
                     "/ground_truth_results/latest.json", "/model_results/latest.json",
                     "/.local/chat_config.json", "/.env", "/assets/../.local/chat_config.json",
                     "/assets/%2e%2e/.local/chat_config.json"]:
            with self.subTest(path=path):
                self.assertEqual(self.client.get(path).status_code,404)
        self.assertEqual(self.client.post("/api/pinn/train",json={}).status_code,404)
        self.service.launch.assert_not_called()
        with self.client.get("/") as response:
            page=response.get_data(as_text=True)
            self.assertNotIn('href="#pinn"',page)
            self.assertNotIn('id="pinn"',page)

    def test_dashboard_has_saved_three_model_results_panel(self):
        with self.client.get("/") as response:
            page=response.get_data(as_text=True)
            self.assertIn('href="#models"',page)
            self.assertIn('id="models"',page)
            self.assertIn('/assets/model-results.js',page)
            self.assertIn('/assets/model-results.css',page)


if __name__ == "__main__":
    unittest.main()
