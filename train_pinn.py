"""Actually train the orbital PINN on catalog initial conditions and physics."""
from __future__ import annotations

import copy
import time

import numpy as np
import torch

from pinn_device import resident_memory_mb
from pinn_model import OrbitalPINN, TIME_SECONDS, physics_loss


def split_objects(rows, seed=2026):
    groups = {}
    for index, row in enumerate(rows):
        signature = tuple(str(row.get(field)) for field in
                          ("EPOCH", "MEAN_MOTION", "ECCENTRICITY", "INCLINATION",
                           "RA_OF_ASC_NODE", "ARG_OF_PERICENTER", "MEAN_ANOMALY", "BSTAR"))
        groups.setdefault(signature, []).append(index)
    keys = list(groups)
    np.random.default_rng(seed).shuffle(keys)
    a, b = int(len(keys)*0.7), int(len(keys)*0.85)
    partitions = {"train": keys[:a], "validation": keys[a:b], "test": keys[b:]}
    return {name: np.array([index for key in selected for index in groups[key]], dtype=int)
            for name, selected in partitions.items()}


def train(initial_states, partitions, hardware, steps=6000, segment_seconds=60,
          seed=2026, report=lambda *args: None, checkpoint=None):
    torch.set_num_threads(hardware["torch_threads"])
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = hardware["selected_device"]
    model = OrbitalPINN(width=hardware["width"]).to(device=device, dtype=torch.float64)
    untrained = copy.deepcopy(model).cpu()
    training = torch.as_tensor(initial_states[partitions["train"]], dtype=torch.float64, device=device)
    validation = torch.as_tensor(initial_states[partitions["validation"]], dtype=torch.float64, device=device)
    if len(training) < 2 or len(validation) < 1:
        raise ValueError("Not enough distinct catalog objects for training and validation")
    tau_max = segment_seconds/TIME_SECONDS
    val_x = validation.repeat_interleave(3, dim=0)
    val_t = torch.tensor([0.2,0.6,0.95], device=device, dtype=torch.float64).repeat(len(validation))[:,None]*tau_max
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=1e-5)
    history = []
    best_loss, best_state, best_step = float("inf"), None, 0
    began = time.perf_counter()
    batch_size = hardware["batch_size"]
    peak_memory = 0.0
    initial_parameters = torch.cat([parameter.detach().cpu().flatten() for parameter in model.parameters()])
    for step in range(steps+1):
        batch_indices = torch.as_tensor(rng.integers(len(training), size=batch_size), device=device)
        x = training[batch_indices]
        tau = torch.as_tensor(rng.uniform(0.02,1.0,(batch_size,1))*tau_max, device=device, dtype=torch.float64)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, details = physics_loss(model, x, tau)
        if not torch.isfinite(loss):
            raise ValueError(f"Nonfinite physics loss at optimizer step {step}")
        if step > 0:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            schedule.step()
        if step % 100 == 0 or step == steps:
            model.eval()
            val_loss, val_details = physics_loss(model, val_x, val_t, create_graph=False)
            value = float(val_loss.detach())
            memory = resident_memory_mb()
            if memory:
                peak_memory = max(peak_memory, memory)
            row = {"step": step, "training_physics_loss": float(loss.detach()), "validation_physics_loss": value,
                   "learning_rate": optimizer.param_groups[0]["lr"], "elapsed_seconds": time.perf_counter()-began,
                   "resident_memory_mb": memory,
                   **{f"validation_{key}": float(v) for key,v in val_details.items()}}
            history.append(row)
            if value < best_loss:
                best_loss, best_step = value, step
                best_state = {key: value.detach().cpu().clone() for key,value in model.state_dict().items()}
                if checkpoint:
                    torch.save({"model_state_dict":best_state,"width":model.width,"depth":model.depth,
                                "best_step":best_step,"validation_physics_loss":best_loss,
                                "segment_seconds":segment_seconds,"seed":seed}, checkpoint)
            report(step/steps, f"PINN step {step:,}/{steps:,} | training {float(loss.detach()):.3e} | validation {value:.3e}", row)
    model.load_state_dict(best_state)
    final_parameters = torch.cat([parameter.detach().cpu().flatten() for parameter in model.parameters()])
    proof = {"parameter_count": int(initial_parameters.numel()), "parameter_l2_change": float(torch.linalg.vector_norm(final_parameters-initial_parameters)),
             "optimizer_steps_executed": steps, "best_step": best_step, "best_validation_physics_loss": best_loss,
             "initial_validation_physics_loss": history[0]["validation_physics_loss"],
             "training_seconds": time.perf_counter()-began, "peak_sampled_resident_memory_mb": peak_memory,
             "training_targets": "Central gravity + J2 differential equation residual only; exact hard initial conditions",
             "future_sgp4_training_samples": 0, "measured_future_training_samples": 0,
             "initial_conditions_source": "Downloaded CelesTrak GP records converted to states by SGP4 at the run start"}
    return model, untrained, history, proof
