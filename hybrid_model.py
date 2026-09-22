"""Supervised SGP4 residual correction, calibrated on earlier real ESA states.

This is a separate neural residual model, not the standalone physics-only PINN.
Prediction accepts only SGP4 states and time; no reference states are needed.
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
from torch import nn


def rtn_basis(r, v):
    r, v = np.asarray(r, float), np.asarray(v, float)
    if r.ndim != 2 or r.shape[1] != 3 or v.shape != r.shape:
        raise ValueError("Expected matching N by 3 position and velocity arrays")
    radius = np.linalg.norm(r, axis=1)
    momentum = np.cross(r, v)
    norm = np.linalg.norm(momentum, axis=1)
    if not np.isfinite(r).all() or not np.isfinite(v).all() or np.any(radius <= 0) or np.any(norm <= 0):
        raise ValueError("Invalid state for RTN basis")
    radial, normal = r / radius[:, None], momentum / norm[:, None]
    return np.stack((radial, np.cross(normal, radial), normal), axis=1)


def features(seconds, period_seconds, time_scale_seconds=5400.0):
    seconds = np.asarray(seconds, dtype=float)
    if seconds.ndim != 1 or not np.isfinite(seconds).all() or np.any(seconds < 0):
        raise ValueError("Times must be finite, nonnegative offsets")
    if not np.isfinite(period_seconds) or period_seconds <= 0 or time_scale_seconds <= 0:
        raise ValueError("Positive period and time scale required")
    phase = 2 * np.pi * seconds / period_seconds
    # Fixed orbital harmonics and elapsed time; no data-dependent feature scaling.
    return np.column_stack((np.sin(phase), np.cos(phase), np.sin(2*phase),
                            np.cos(2*phase), seconds/time_scale_seconds))


class ResidualCorrector(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.width = width
        self.network = nn.Sequential(nn.Linear(5, width), nn.Tanh(),
                                     nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 6))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, x, seconds):
        # Retain the exact common GP-derived initial r and v in all three models.
        gate = -torch.expm1(-(seconds[:, None]/60.0).square())
        return self.network(x) * gate


def residual_targets(data):
    basis = rtn_basis(data["sgp4_r"], data["sgp4_v"])
    dr = np.asarray(data["reference_r"]) - data["sgp4_r"]
    dv = np.asarray(data["reference_v"]) - data["sgp4_v"]
    targets = np.concatenate((np.einsum("nij,nj->ni", basis, dr),
                              np.einsum("nij,nj->ni", basis, dv)), axis=1)
    if not np.isfinite(targets).all():
        raise ValueError("Nonfinite supervised residual target")
    return targets


def train_corrector(training, validation, hardware, period_seconds, steps=3000,
                    seed=2026, checkpoint=None, report=lambda *args: None):
    """Only training and validation arrays enter this function; no test argument."""
    if type(steps) is not int or not 100 <= steps <= 30000:
        raise ValueError("Use 100 to 30000 optimizer steps")
    train_t = np.asarray(training["seconds"], float)
    val_t = np.asarray(validation["seconds"], float)
    if len(train_t) < 10 or len(val_t) < 5 or train_t.min() <= 0 or train_t.max() >= val_t.min():
        raise ValueError("Need chronological, nonoverlapping positive-time training/validation data")
    torch.set_num_threads(hardware["torch_threads"])
    torch.manual_seed(seed)
    device = hardware["selected_device"]
    model = ResidualCorrector().to(device=device, dtype=torch.float64)
    initial = torch.cat([p.detach().cpu().flatten() for p in model.parameters()])
    train_y, val_y = residual_targets(training), residual_targets(validation)
    scale = np.maximum(np.sqrt(np.mean(train_y**2, axis=0)), [0.001]*3 + [0.000001]*3)
    tensor = lambda x: torch.as_tensor(x, dtype=torch.float64, device=device)
    x, t, y = tensor(features(train_t, period_seconds)), tensor(train_t), tensor(train_y/scale)
    vx, vt, vy = tensor(features(val_t, period_seconds)), tensor(val_t), tensor(val_y/scale)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps, eta_min=0.00001)
    best_loss, best_state, best_step = float("inf"), None, 0
    history, began = [], time.perf_counter()
    for step in range(steps+1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = (model(x, t)-y).square().mean()
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite correction loss")
        if step:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            scheduler.step()
        if step % 100 == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                training_loss = float((model(x, t)-y).square().mean())
                validation_loss = float((model(vx, vt)-vy).square().mean())
            row = {"step":step, "training_loss":training_loss, "validation_loss":validation_loss,
                   "elapsed_seconds":time.perf_counter()-began, "learning_rate":optimizer.param_groups[0]["lr"]}
            history.append(row)
            if validation_loss < best_loss:
                best_loss, best_step = validation_loss, step
                best_state = copy.deepcopy({k:v.detach().cpu() for k,v in model.state_dict().items()})
            report(step/steps, f"Correction step {step:,}/{steps:,} | train {training_loss:.4g} | validation {validation_loss:.4g}", row)
    model.load_state_dict(best_state)
    final = torch.cat([p.detach().cpu().flatten() for p in model.parameters()])
    configuration = {"width":model.width, "scale":scale.tolist(), "period_seconds":float(period_seconds),
                     "time_scale_seconds":5400.0, "best_step":best_step, "seed":seed,
                     "scope":"Sentinel-1D GP arc only; not a debris-catalog correction model"}
    if checkpoint:
        torch.save({"model_state_dict":best_state, **configuration}, checkpoint)
    proof = {"parameter_count":int(initial.numel()), "parameter_l2_change":float(torch.linalg.vector_norm(final-initial)),
             "optimizer_steps_executed":steps, "best_step":best_step, "training_seconds":time.perf_counter()-began,
             "initial_validation_loss":history[0]["validation_loss"], "best_validation_loss":best_loss,
             "training_samples":len(train_t), "validation_samples":len(val_t), "test_samples_received":0,
             "training_seconds_min":float(train_t.min()), "training_seconds_max":float(train_t.max()),
             "validation_seconds_min":float(val_t.min()), "validation_seconds_max":float(val_t.max()),
             "training_targets":"Earlier independent ESA minus SGP4 r/v, in SGP4 RTN basis",
             "objective":"Mean squared six-component residual divided by training-only RMS scales",
             "model_selection":"Lowest validation loss, checked every 100 steps; fixed hyperparameters before test evaluation",
             "classification":"Supervised neural residual correction, not the physics-only standalone PINN",
             "configuration":configuration}
    return model, configuration, history, proof


def predict_corrected(model, configuration, seconds, sgp4_r, sgp4_v, device="cpu"):
    basis = rtn_basis(sgp4_r, sgp4_v)
    x = features(seconds, configuration["period_seconds"], configuration["time_scale_seconds"])
    if len(x) != len(basis):
        raise ValueError("Time and state counts differ")
    model.eval()
    with torch.no_grad():
        correction = model(torch.as_tensor(x, dtype=torch.float64, device=device),
                           torch.as_tensor(seconds, dtype=torch.float64, device=device)).cpu().numpy()
    correction *= np.asarray(configuration["scale"])
    dr = np.einsum("nij,ni->nj", basis, correction[:, :3])
    dv = np.einsum("nij,ni->nj", basis, correction[:, 3:])
    return {"r":np.asarray(sgp4_r)+dr, "v":np.asarray(sgp4_v)+dv}


def load_corrector(path):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    model = ResidualCorrector(saved["width"]).double().eval()
    model.load_state_dict(saved["model_state_dict"])
    return model, {k:v for k,v in saved.items() if k != "model_state_dict"}
