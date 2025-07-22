import os
import torch

from ikflow.config import DATASET_DIR, ALL_DATASET_TAGS, DATASET_TAG_NON_SELF_COLLIDING
from ikflow.utils import get_dataset_filepaths

# Point this at your specific dataset folder name:
dataset_name = "ur5e_custom_non-self-colliding"
base = os.path.join(DATASET_DIR, dataset_name)

# Figure out exactly where torch saved things (handles the __tag0=… suffix)
samples_tr_fp, poses_tr_fp, samples_te_fp, poses_te_fp, info_fp = \
    get_dataset_filepaths(base, [DATASET_TAG_NON_SELF_COLLIDING])

# 1) Load
samples_tr   = torch.load(samples_tr_fp)
poses_tr     = torch.load(poses_tr_fp)

# 2) Basic shapes and first entries
print("samples_tr shape:", samples_tr.shape)  # e.g. (2000000, 6)
print("poses_tr   shape:", poses_tr.shape)    # e.g. (2000000, 7)
print("\nFirst joint configuration:\n", samples_tr[0])
print("Corresponding end-effector pose:\n", poses_tr[0], "\n")

# 3) Compute per-column mean & std
joint_mean = samples_tr.mean(dim=0)
joint_std  = samples_tr.std(dim=0)
pose_mean  = poses_tr.mean(dim=0)
pose_std   = poses_tr.std(dim=0)

# 4) Print out the normalization constants
print("JOINT_MEAN =", joint_mean.tolist())
print("JOINT_STD  =", joint_std.tolist())
print("POSE_MEAN  =", pose_mean.tolist())
print("POSE_STD   =", pose_std.tolist())
