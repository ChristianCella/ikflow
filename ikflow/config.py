"""Global configuration"""

import os
import torch

from jrl.config import DEVICE

# Expose `device` so downstream imports (e.g. in lt_data.py) work correctly
device = DEVICE

DEFAULT_TORCH_DTYPE = torch.float32

print(f"ikflow/config.py | Using device: '{DEVICE}'")

# ------------------------------------------------------------------------------
# Paths for various caches/logs. You can still override the cache root via
# the IKFLOW_CACHE_DIR env var if you like—but datasets will always go under
# `ikflow/datasets/` inside your repo.
# ------------------------------------------------------------------------------

# Base directory of this ikflow package (i.e. <repo>/ikflow/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Other caches (wandb, models, training logs) still use the default cache dir
DEFAULT_DATA_DIR = os.getenv(
    "IKFLOW_CACHE_DIR",
    os.path.join(os.path.expanduser("~"), ".cache", "ikflow")
)
WANDB_CACHE_DIR   = os.path.join(DEFAULT_DATA_DIR,    "wandb")
TRAINING_LOGS_DIR = os.path.join(DEFAULT_DATA_DIR,    "training_logs")
MODELS_DIR        = os.path.join(DEFAULT_DATA_DIR,    "models")

# ------------------------------------------------------------------------------
# Datasets directory: placed directly under the ikflow package root
# ------------------------------------------------------------------------------

DATASET_DIR = os.path.join(BASE_DIR, "datasets")
print(f"ikflow will store datasets under: {DATASET_DIR}")

# ------------------------------------------------------------------------------
# Dataset tags
# ------------------------------------------------------------------------------

DATASET_TAG_NON_SELF_COLLIDING = "non-self-colliding"
ALL_DATASET_TAGS               = [DATASET_TAG_NON_SELF_COLLIDING]

# ------------------------------------------------------------------------------
# Training hyperparameters / limits
# ------------------------------------------------------------------------------

# Maximum absolute value for joint-angle side of the network when using sigmoid activation
SIGMOID_SCALING_ABS_MAX = 1.0
