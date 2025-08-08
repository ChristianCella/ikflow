#!/usr/bin/env python
import os, sys
from datetime import datetime

import torch
from jrl.robots import Robot
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.trainer import Trainer
from pytorch_lightning import seed_everything
import wandb

from ikflow.config import DATASET_TAG_NON_SELF_COLLIDING, ALL_DATASET_TAGS, WANDB_CACHE_DIR
import ikflow.config as ikcfg
from ikflow.model import IkflowModelParameters
from ikflow.ikflow_solver import IKFlowSolver
from ikflow.training.lt_model import IkfLitModel
from ikflow.training.lt_data import IkfLitDataset
from ikflow.training.training_utils import get_checkpoint_dir
from ikflow.utils import boolean_string, non_private_dict, get_wandb_project

# === USER CONFIGURATION ===
ROBOT_NAME            = "ur5e_custom"
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
print(f"Base directory: {base_dir}")
sys.path.append(base_dir)
URDF_PATH = os.path.join(base_dir, "ur5e_utils_mujoco/ur5e/patched_ur5e.urdf")
DISABLE_WANDB         = True

# Model hyperparameters
COUPLING_LAYER        = "glow"
RNVP_CLAMP            = 2.5
SOFTFLOW_NOISE_SCALE  = 0.001
SOFTFLOW_ENABLED      = True
NB_NODES              = 6
DIM_LATENT_SPACE      = 8
COEFF_FN_CONFIG       = 3
COEFF_FN_INTERNAL_SIZE= 1024
Y_NOISE_SCALE         = 1e-7
ZEROS_NOISE_SCALE     = 1e-3
SIGMOID_ON_OUTPUT     = False

# Training parameters
OPTIMIZER             = "adamw"
BATCH_SIZE            = 256
GAMMA                 = 0.9794578299341784
LEARNING_RATE         = 1e-5
LAMBDA                = 1.0
WEIGHT_DECAY          = 1e-4
STEP_LR_EVERY         = int(2e6 / BATCH_SIZE)
GRADIENT_CLIP_VAL     = 1.0
MAX_EPOCHS            = 300

# Logging / checkpointing
VAL_SET_SIZE          = 1000
LOG_EVERY             = 20000
CHECKPOINT_EVERY      = 20000
DATASET_TAGS          = [DATASET_TAG_NON_SELF_COLLIDING]
RUN_DESCRIPTION       = None
DISABLE_PROGRESS_BAR  = False
# =======================

# Fix seed for reproducibility
SEED = 0
seed_everything(SEED, workers=True)

# --- Robot setup ---
active_joints = [
    "shoulder_pan_joint","shoulder_lift_joint","elbow_joint",
    "wrist_1_joint","wrist_2_joint","wrist_3_joint"
]
robot = Robot(
    name=ROBOT_NAME,
    urdf_filepath=URDF_PATH,
    active_joints=active_joints,
    base_link="base_link",
    end_effector_link_name="wrist_3_link",
    ignored_collision_pairs=[],
    collision_capsules_by_link=None,
)
print(f"Loaded robot '{robot.name}' (DOF={robot.ndof})")

# --- DataModule ---
data_module = IkfLitDataset(
    robot_name   = robot.name,
    batch_size   = BATCH_SIZE,
    val_set_size = VAL_SET_SIZE,
    dataset_tags = DATASET_TAGS,
)

# --- Hyperparameters object ---
base_hparams = IkflowModelParameters()
base_hparams.coupling_layer           = COUPLING_LAYER
base_hparams.rnvp_clamp               = RNVP_CLAMP
base_hparams.softflow_noise_scale     = SOFTFLOW_NOISE_SCALE
base_hparams.softflow_enabled         = SOFTFLOW_ENABLED
base_hparams.nb_nodes                 = NB_NODES
base_hparams.dim_latent_space         = DIM_LATENT_SPACE
base_hparams.coeff_fn_config          = COEFF_FN_CONFIG
base_hparams.coeff_fn_internal_size   = COEFF_FN_INTERNAL_SIZE
base_hparams.y_noise_scale            = Y_NOISE_SCALE
base_hparams.zeros_noise_scale        = ZEROS_NOISE_SCALE
base_hparams.sigmoid_on_output        = SIGMOID_ON_OUTPUT
base_hparams.run_description          = RUN_DESCRIPTION
print(base_hparams)

# --- WandB logger (optional) ---
wandb_logger = None
if not DISABLE_WANDB:
    entity, project = get_wandb_project()
    cfg = {"robot": ROBOT_NAME}
    cfg.update(non_private_dict({
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        # ... add any other hyperparams you want to log
    }))
    data_module.add_dataset_hashes_to_cfg(cfg)
    wandb.init(
        entity=entity,
        project=project,
        notes=RUN_DESCRIPTION,
        config=cfg,
    )
    wandb_logger = WandbLogger(log_model="all", save_dir=WANDB_CACHE_DIR)
    wandb.run.tags = wandb.run.tags + tuple(DATASET_TAGS)

# --- Build solver & Lightning model ---
ik_solver = IKFlowSolver(base_hparams, robot)
model = IkfLitModel(
    ik_solver        = ik_solver,
    base_hparams     = base_hparams,
    learning_rate    = LEARNING_RATE,
    checkpoint_every = CHECKPOINT_EVERY,
    log_every        = LOG_EVERY,
    gradient_clip    = GRADIENT_CLIP_VAL,
    lambd            = LAMBDA,
    gamma            = GAMMA,
    step_lr_every    = STEP_LR_EVERY,
    weight_decay     = WEIGHT_DECAY,
    optimizer_name   = OPTIMIZER,
    sigmoid_on_output= SIGMOID_ON_OUTPUT,
)

# --- Checkpoint callback ---
ckpt_dir = get_checkpoint_dir(ROBOT_NAME)
checkpoint_callback = ModelCheckpoint(
    dirpath                 = ckpt_dir,
    every_n_epochs          = 25,
    save_on_train_epoch_end = True,
    save_top_k              = -1,
    filename                = "weights-{epoch}",
)
if not DISABLE_WANDB:
    wandb.config.update({"checkpoint_directory": ckpt_dir})

# --- Trainer ---
trainer = Trainer(
    logger                   = False,
    callbacks                = [checkpoint_callback],
    check_val_every_n_epoch  = 1,
    devices                  = [torch.cuda.current_device()] if torch.cuda.is_available() else None,
    accelerator              = "gpu" if torch.cuda.is_available() else "cpu",
    log_every_n_steps        = LOG_EVERY,
    max_epochs               = MAX_EPOCHS,
    enable_progress_bar      = not DISABLE_PROGRESS_BAR,
)

# --- Launch training ---
trainer.fit(model, data_module)
