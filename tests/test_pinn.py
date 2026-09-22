"""Numerical unit tests. Synthetic geometries here are not experiment outputs."""
import unittest
from unittest.mock import patch
import numpy as np
import torch

from collision_probability import apply_covariances, gaussian_disk_probability
from pinn_evaluation import pair_classification
from pinn_model import OrbitalPINN, acceleration, physics_loss, predict_step, rollout
from pinn_screening import refine_batch
from train_pinn import split_objects


class PINNTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(5)
        self.model=OrbitalPINN(16,2).double()
        self.initial=torch.tensor([[1.,0,0,0,1,0],[0,1.,0,-1,0,0]],dtype=torch.float64)

    def test_initial_position_and_velocity_enforced_exactly(self):
        result=self.model(self.initial,torch.zeros((2,1),dtype=torch.float64))
        torch.testing.assert_close(result,self.initial,rtol=0,atol=0)

    def test_physics_loss_backpropagates_into_network(self):
        loss,_=physics_loss(self.model,self.initial,torch.full((2,1),.05,dtype=torch.float64))
        loss.backward()
        self.assertTrue(np.isfinite(float(loss.detach())))
        self.assertGreater(sum(float(p.grad.norm()) for p in self.model.parameters() if p.grad is not None),0)

    def test_gravity_symmetry_and_sign(self):
        a=acceleration(torch.tensor([[1.,0,0],[-1.,0,0],[0,0,1.]],dtype=torch.float64)).numpy()
        np.testing.assert_allclose(a[0],-a[1],atol=1e-12)
        self.assertLess(a[0,0],0)
        self.assertLess(a[2,2],0)

    def test_prediction_and_rollout_shapes(self):
        x=self.initial.numpy()
        np.testing.assert_allclose(predict_step(self.model,x,0),x,rtol=0,atol=0)
        trajectory=rollout(self.model,x,np.array([0.,30.,60.]))
        self.assertEqual(trajectory.shape,(2,3,6))
        self.assertTrue(np.isfinite(trajectory).all())

    def test_duplicate_orbits_stay_in_one_partition(self):
        rows=[{"EPOCH":"2026-09-10T00:00:00","MEAN_MOTION":i} for i in range(30)]
        rows.append(dict(rows[0]))
        splits=split_objects(rows)
        membership={int(index):name for name,values in splits.items() for index in values}
        self.assertEqual(membership[0],membership[30])
        self.assertEqual(len(membership),31)

    @patch("pinn_screening.predict_step")
    def test_neural_refinement_finds_fast_between_sample_minimum(self,predict):
        def linear(model,initial,seconds,device):
            result=initial.copy()
            result[:,:3]+=np.asarray(seconds)[:,None]*initial[:,3:]
            return result
        predict.side_effect=linear
        a=np.array([[-3.,0,0,1,0,0]])
        b=np.array([[3.,0,0,-1,0,0]])
        t,d,_,_=refine_batch(None,a,b,10.,"cpu")
        self.assertAlmostEqual(t[0],3.,places=4)
        self.assertLess(d[0],0.2)

    def test_reference_classification_is_explicitly_not_observations(self):
        events=[{"object1_id":1,"object2_id":2}]
        ref=[{"object1_id":1,"object2_id":3}]
        metric,_=pair_classification(events,ref,[1,2,3],[])
        self.assertEqual(metric["confusion_matrix"],{"true_positive":0,"false_positive":1,"false_negative":1,"true_negative":1})
        self.assertEqual(metric["recall"],0)
        self.assertIn("not observed",metric["interpretation"])

    def test_missing_covariance_never_produces_probability(self):
        events=[{"collision_probability":None}]
        result=apply_covariances(events)
        self.assertEqual(result["status"],"unavailable")
        self.assertIsNone(events[0]["collision_probability"])

    def test_probability_integral_matches_isotropic_analytic_case(self):
        value=gaussian_disk_probability([0,0],np.eye(2)*4,1)
        self.assertAlmostEqual(value,1-np.exp(-1/8),places=10)

    def test_invalid_covariance_rejected(self):
        with self.assertRaises(ValueError):gaussian_disk_probability([0,0],[[1,2],[2,1]],1)


if __name__=="__main__":unittest.main()
