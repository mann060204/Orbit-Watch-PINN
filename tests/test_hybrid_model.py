"""Offline numerical tests with synthetic fixtures, never experiment outputs.

The circular states and prescribed residuals below exist only to exercise the
optimizer and coordinate transformations. They are not ESA or CelesTrak data.
"""
import inspect
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from hybrid_model import (
    ResidualCorrector, load_corrector, predict_corrected, residual_targets,
    rtn_basis, train_corrector,
)


PERIOD_SECONDS = 5400.0
HARDWARE = {"selected_device": "cpu", "torch_threads": 1}


def synthetic_unit_fixture(seconds):
    """A prescribed circular arc and smooth RTN residual for unit tests only."""
    seconds = np.asarray(seconds, dtype=float)
    angle = seconds * 2 * np.pi / PERIOD_SECONDS
    omega = 2 * np.pi / PERIOD_SECONDS
    r = 7000 * np.column_stack((np.cos(angle), np.sin(angle), np.zeros_like(angle)))
    v = 7000 * omega * np.column_stack((-np.sin(angle), np.cos(angle), np.zeros_like(angle)))
    basis = rtn_basis(r, v)
    gate = -np.expm1(-(seconds[:, None] / 60.0) ** 2)
    dr_rtn = gate * np.array([0.02, -0.06, 0.015])
    dv_rtn = gate * np.array([0.00001, 0.00002, -0.000005])
    return {
        "seconds": seconds,
        "sgp4_r": r,
        "sgp4_v": v,
        "reference_r": r + np.einsum("nij,ni->nj", basis, dr_rtn),
        "reference_v": v + np.einsum("nij,ni->nj", basis, dv_rtn),
    }


class HybridModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.training = synthetic_unit_fixture(np.arange(60.0, 601.0, 60.0))
        cls.validation = synthetic_unit_fixture(np.arange(660.0, 901.0, 60.0))
        cls.temporary = TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.checkpoint = Path(cls.temporary.name) / "synthetic_unit_test_model.pt"
        cls.model, cls.configuration, cls.history, cls.proof = train_corrector(
            cls.training, cls.validation, HARDWARE, PERIOD_SECONDS,
            steps=100, seed=2026, checkpoint=cls.checkpoint,
        )

    def test_rtn_basis_projection_has_correct_orientation_and_inverse(self):
        r = np.array([[7000.0, 0, 0], [0, 7000.0, 0], [1, 2, 3]])
        v = np.array([[0, 7.5, 0], [-7.5, 0, 0], [-2, 1, 1]])
        basis = rtn_basis(r, v)
        np.testing.assert_allclose(basis[0], np.eye(3), rtol=0, atol=1e-15)
        np.testing.assert_allclose(basis[1], [[0, 1, 0], [-1, 0, 0], [0, 0, 1]], rtol=0, atol=1e-15)
        np.testing.assert_allclose(np.linalg.det(basis), 1.0, atol=1e-14)
        vectors = np.array([[0.1, -0.2, 0.3], [4.0, 2, -1], [0.4, 0.9, -0.7]])
        projected = np.einsum("nij,nj->ni", basis, vectors)
        reconstructed = np.einsum("nij,ni->nj", basis, projected)
        np.testing.assert_allclose(reconstructed, vectors, rtol=1e-14, atol=1e-14)

    def test_residual_targets_match_prescribed_fixture_rtn_offsets(self):
        fixture = synthetic_unit_fixture([0, 60, 600])
        expected = -np.expm1(-(fixture["seconds"][:, None] / 60.0) ** 2) * np.array(
            [0.02, -0.06, 0.015, 0.00001, 0.00002, -0.000005]
        )
        np.testing.assert_allclose(residual_targets(fixture), expected, rtol=1e-8, atol=1e-12)

    def test_invalid_rtn_states_are_rejected(self):
        for r, v in [([[0, 0, 0]], [[0, 1, 0]]),
                     ([[1, 0, 0]], [[2, 0, 0]]),
                     ([[np.nan, 0, 0]], [[0, 1, 0]]),
                     ([1, 0, 0], [0, 1, 0])]:
            with self.subTest(r=r, v=v), self.assertRaises(ValueError):
                rtn_basis(r, v)

    def test_initial_state_is_exact_with_trained_and_nonzero_models(self):
        fixture = synthetic_unit_fixture([0, 300])
        nonzero = ResidualCorrector(width=8).double().eval()
        with torch.no_grad():
            for parameter in nonzero.parameters():
                parameter.fill_(0.125)
        for model in [self.model, nonzero]:
            with self.subTest(model_width=model.width):
                result = predict_corrected(model, self.configuration, fixture["seconds"],
                                           fixture["sgp4_r"], fixture["sgp4_v"])
                np.testing.assert_array_equal(result["r"][0], fixture["sgp4_r"][0])
                np.testing.assert_array_equal(result["v"][0], fixture["sgp4_v"][0])
                self.assertGreater(np.linalg.norm(result["r"][1] - fixture["sgp4_r"][1]), 0)
                self.assertGreater(np.linalg.norm(result["v"][1] - fixture["sgp4_v"][1]), 0)

    def test_actual_optimizer_changes_weights_and_reduces_fixture_validation_loss(self):
        self.assertEqual(self.proof["optimizer_steps_executed"], 100)
        self.assertGreater(self.proof["parameter_l2_change"], 0)
        self.assertGreater(float(torch.linalg.vector_norm(self.model.network[-1].weight.detach())), 0)
        self.assertLess(self.proof["best_validation_loss"], self.proof["initial_validation_loss"])
        self.assertEqual([row["step"] for row in self.history], [0, 100])
        self.assertEqual(self.proof["training_samples"], 10)
        self.assertEqual(self.proof["validation_samples"], 5)

    def test_checkpoint_replays_corrected_states_without_reference_inputs(self):
        reloaded, configuration = load_corrector(self.checkpoint)
        self.assertEqual(configuration, self.configuration)
        fixture = synthetic_unit_fixture([0, 75, 750, 1200])
        inputs = (fixture["seconds"], fixture["sgp4_r"], fixture["sgp4_v"])
        expected = predict_corrected(self.model, self.configuration, *inputs)
        actual = predict_corrected(reloaded, configuration, *inputs)
        np.testing.assert_array_equal(actual["r"], expected["r"])
        np.testing.assert_array_equal(actual["v"], expected["v"])

    def test_malformed_chronological_partitions_fail_before_training(self):
        for training_seconds, validation_seconds in [
            (np.arange(0.0, 600.0, 60.0), np.arange(660.0, 901.0, 60.0)),
            (np.arange(-60.0, 540.0, 60.0), np.arange(660.0, 901.0, 60.0)),
            (np.arange(60.0, 601.0, 60.0), np.arange(600.0, 841.0, 60.0)),
            (np.arange(60.0, 601.0, 60.0), np.arange(360.0, 601.0, 60.0)),
            (np.arange(60.0, 541.0, 60.0), np.arange(660.0, 901.0, 60.0)),
            (np.arange(60.0, 601.0, 60.0), np.arange(660.0, 841.0, 60.0)),
        ]:
            with self.subTest(training=training_seconds, validation=validation_seconds):
                with self.assertRaisesRegex(ValueError, "chronological"):
                    train_corrector(synthetic_unit_fixture(training_seconds),
                                    synthetic_unit_fixture(validation_seconds),
                                    HARDWARE, PERIOD_SECONDS, steps=100)

    def test_training_api_accepts_no_holdout_partition(self):
        parameters = inspect.signature(train_corrector).parameters
        self.assertIn("training", parameters)
        self.assertIn("validation", parameters)
        self.assertNotIn("test", parameters)
        self.assertNotIn("holdout", parameters)
        self.assertFalse(any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()))
        with self.assertRaises(TypeError):
            train_corrector(self.training, self.validation, HARDWARE, PERIOD_SECONDS,
                            steps=100, test=self.validation)
        self.assertEqual(self.proof["test_samples_received"], 0)


if __name__ == "__main__":
    unittest.main()
