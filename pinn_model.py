"""A physics-informed short-time orbital flow map, trained without future tracks.

Real CelesTrak-derived initial states are the only data constraints. A neural
correction to a Taylor trial solution is trained through automatic differentiation
against central gravity + J2. No SGP4 future labels enter the training objective.
"""
from __future__ import annotations

import math
import numpy as np
import torch
from torch import nn

LENGTH_KM = 7000.0
MU_KM3_S2 = 398600.8  # WGS72, matching the GP/SGP4 convention.
EARTH_RADIUS_KM = 6378.135
J2 = 0.001082616
TIME_SECONDS = math.sqrt(LENGTH_KM**3 / MU_KM3_S2)
SPEED_KM_S = LENGTH_KM / TIME_SECONDS


def acceleration(position):
    """Dimensionless central gravity plus J2 in a quasi-inertial equatorial frame."""
    radius2 = (position**2).sum(dim=-1, keepdim=True)
    radius = torch.sqrt(radius2)
    gravity = -position / radius**3
    z2_over_r2 = position[..., 2:3]**2 / radius2
    factors = torch.cat((5*z2_over_r2-1, 5*z2_over_r2-1, 5*z2_over_r2-3), dim=-1)
    oblateness = 1.5 * J2 * (EARTH_RADIUS_KM/LENGTH_KM)**2 * position * factors / radius**5
    return gravity + oblateness


class OrbitalPINN(nn.Module):
    def __init__(self, width=48, depth=3):
        super().__init__()
        layers = []
        features = 6
        for _ in range(depth):
            layer = nn.Linear(features, width)
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
            layers.extend((layer, nn.Tanh()))
            features = width
        head = nn.Linear(features, 6)
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)
        layers.append(head)
        self.network = nn.Sequential(*layers)
        self.width, self.depth = width, depth

    def forward(self, initial_state, tau):
        r0, v0 = initial_state[..., :3], initial_state[..., 3:]
        radius = torch.linalg.vector_norm(r0, dim=-1, keepdim=True)
        features = torch.cat((radius-1, (r0*v0).sum(-1, keepdim=True)/radius,
                              (v0*v0).sum(-1, keepdim=True)-1, tau,
                              r0[..., 2:3]/radius, v0[..., 2:3]), dim=-1)
        coefficients = self.network(features)
        axis = torch.zeros_like(r0)
        axis[..., 2] = 1
        correction_r = coefficients[..., 0:1]*r0 + coefficients[..., 1:2]*v0 + coefficients[..., 2:3]*axis
        correction_v = coefficients[..., 3:4]*r0 + coefficients[..., 4:5]*v0 + coefficients[..., 5:6]*axis
        a0 = acceleration(r0)
        r = r0 + tau*v0 + 0.5*tau**2*a0 + tau**3*correction_r
        v = v0 + tau*a0 + tau**2*correction_v
        return torch.cat((r, v), dim=-1)


def physics_loss(model, initial_states, tau, create_graph=True):
    tau = tau.detach().clone().requires_grad_(True)
    output = model(initial_states, tau)
    derivative = torch.cat([torch.autograd.grad(output[:, component].sum(), tau,
                                               create_graph=create_graph, retain_graph=True)[0]
                            for component in range(6)], dim=1)
    kinematic = derivative[:, :3] - output[:, 3:]
    dynamic = derivative[:, 3:] - acceleration(output[:, :3])
    # Remove trivial small-time scaling so learning cannot win by predicting the
    # exact initial condition alone. Collocation always uses strictly positive tau.
    kinematic_scaled = kinematic / tau.square()
    dynamic_scaled = dynamic / tau
    loss = kinematic_scaled.square().mean() + dynamic_scaled.square().mean()
    return loss, {
        "scaled_kinematic_mse": kinematic_scaled.square().mean().detach(),
        "scaled_dynamic_mse": dynamic_scaled.square().mean().detach(),
        "kinematic_rms_km_s": kinematic.square().mean().sqrt().detach()*SPEED_KM_S,
        "dynamic_rms_km_s2": dynamic.square().mean().sqrt().detach()*LENGTH_KM/TIME_SECONDS**2,
    }


def normalize_states(positions, velocities):
    return np.concatenate((np.asarray(positions)/LENGTH_KM, np.asarray(velocities)/SPEED_KM_S), axis=-1)


def predict_step(model, initial_states, seconds, device="cpu", batch_size=4096):
    """Return normalized next states; seconds may be scalar or one value per row."""
    initial_states = np.asarray(initial_states, dtype=np.float64)
    times = np.broadcast_to(np.asarray(seconds, dtype=float), (len(initial_states),))
    result = np.empty_like(initial_states)
    with torch.no_grad():
        for start in range(0, len(initial_states), batch_size):
            end = start + batch_size
            x = torch.as_tensor(initial_states[start:end], dtype=torch.float64, device=device)
            tau = torch.as_tensor(np.array(times[start:end, None]/TIME_SECONDS), dtype=torch.float64, device=device)
            result[start:end] = model(x, tau).cpu().numpy()
    return result


def rollout(model, initial_states, seconds, device="cpu"):
    states = np.empty((len(initial_states), len(seconds), 6), dtype=np.float64)
    states[:, 0] = initial_states
    for index in range(len(seconds)-1):
        states[:, index+1] = predict_step(model, states[:, index], seconds[index+1]-seconds[index], device)
    return states
