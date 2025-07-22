from datetime import datetime
import os

import numpy as np
import torch
import wandb

from ikflow import config


def get_softflow_noise(x: torch.Tensor, softflow_noise_scale: float):
    """
    Return noise and noise magnitude for softflow. See https://arxiv.org/abs/2006.04604
    Returns:
        c: (batch_sz x 1) tensor of magnitudes
        eps: (batch_sz x dim_x) tensor of noise
    """
    dim_x = x.shape[1]
    c = torch.rand_like(x[:, 0]).unsqueeze(1)
    eps = torch.randn_like(x) * torch.cat([c] * dim_x, dim=1) * softflow_noise_scale
    return c, eps


def _np_to_str(x: np.ndarray) -> str:
    assert x.ndim == 1
    return str([round(x_i, 4) for x_i in x.tolist()])


def _datetime_str() -> str:
    """
    Return a Windows‐compatible timestamp string for checkpoint directories.
    Original '%-M' is unsupported on Windows, so we use '%M' and include a separator.
    Example result: 'Jul.12.2025_03-35PM'
    """
    now = datetime.now()
    return now.strftime("%b.%d.%Y_%I-%M%p")


def get_checkpoint_dir(robot_name: str) -> str:
    """
    Create (and on-demand make) a new checkpoint directory path,
    placing it in a 'weights/' folder alongside the 'datasets/' folder
    inside your ikflow package.
    
    Resulting layout:
      <repo>/ikflow/ikflow/
        ├─ datasets/
        └─ weights/
            └─ <robot_name>--<timestamp>--(wandb-run-<name>)/
    """
    # Determine the parent directory containing 'datasets/'
    base_dir = os.path.dirname(config.DATASET_DIR)

    # Create (if needed) a sibling 'weights' directory
    weights_root = os.path.join(base_dir, "weights")
    os.makedirs(weights_root, exist_ok=True)

    # Build a run-specific subfolder name
    ts = _datetime_str()
    if wandb.run is not None:
        run_suffix = f"--wandb-run-{wandb.run.name}"
    else:
        run_suffix = ""
    run_name = f"{robot_name}--{ts}{run_suffix}"

    # Full checkpoint directory path
    full_dir = os.path.join(weights_root, run_name)
    os.makedirs(full_dir, exist_ok=True)
    return full_dir
