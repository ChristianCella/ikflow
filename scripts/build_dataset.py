"""
build_dataset.py — Automatically generate robot joint-angle datasets with example outputs

This script creates training and test datasets of joint configurations and corresponding end-effector poses for a given robot (loaded via URDF). Everything is hard-coded at the top so you can simply run:

    python scripts/build_dataset.py

and you’ll see output like:

    Building dataset for robot: ur5e_custom
    100%|…| 15000/15000 [...]      # test set sampling
    100%|…| 2000000/2000000 [...]   # training set sampling

    min, max, mean, std — for 'samples_tr':
      col_0:  -6.2788, 6.2788, -0.0072, 3.6257  # joint 0
      …
      col_5:  -6.2788, 6.2788,  0.0028, 3.6258  # joint 5

    min, max, mean, std — for 'poses_tr':
      col_0:  -0.9259, 0.9244, 0.0001, 0.3087  # x-coordinate
      …
      col_6:  -0.9999, 1.0000, 0.0005, 0.4998  # quaternion z

    (similarly for 'samples_te' and 'poses_te')
    Saved dataset with 2000000 samples in 43.30 seconds
    Summary info on all saved datasets:
      ur5e  ur5e_custom_non-self-colliding  69.0626

--- How the dataset is created ---
1. **Robot instantiation**
   - Loads `ur5e_custom` from `URDF_PATH` (e.g. `C:\...\ur5e.urdf`).
   - Reads joint limits such as [-6.283, +6.283] for revolute joints.

2. **Sampling configurations**
   - **Test set**: draws 15,000 joint-angle vectors uniformly in each limit ±ε,
   - **Training set**: draws 2,000,000 samples similarly.
   - With `ONLY_NON_SELF_COLLIDING=True`, any self-colliding pose is discarded and re-sampled until reaching the desired size.

3. **Forward kinematics**
   - Computes end-effector pose (x,y,z, qw,qx,qy,qz) for each sample.

4. **Tensor conversion & sanity checks**
   - Converts to `torch.float32` tensors.
   - Verifies each column has std > 0.001. E.g., joint std ≈3.6257 rad, pose x std ≈0.3087.
   - Asserts all angles respect joint limits.

5. **File outputs** under `DATASET_DIR/ur5e_custom_non-self-colliding/`:
   - `samples_tr.pt`, `poses_tr.pt`
   - `samples_te.pt`, `poses_te.pt`
   - `info.txt` (human-readable stats)

--- Console output explained (with your values) ---
- **Sampling progress bars**: show completion of 15k and 2M samples.
- **Statistics blocks**:
  - For `'samples_tr'`, joint 0 ranged [−6.2788, +6.2788], mean ≈ −0.0072, std ≈ 3.6257.
  - For `'poses_tr'`, x-coordinate ranged [−0.9259, +0.9244], mean ≈ 0.0001, std ≈ 0.3087.
  - Similar stats for the test set.
- **Timing**: You saw 43.30 seconds for 2M samples.

**sum_joint_range = 69.0626** means:
> Σ over 6 joints of (max − min) ≈ 69.06 rad covered in training set.
Max possible 6×(2π)≈37.70? Actually each joint ±2π gives 4π span≈12.566, so 6×12.566≈75.4 max; you covered ~69.06 after collision filtering.

--- Training vs. Test split ---
- **Training**: 2 000 000 samples for model fitting (`samples_tr.pt`).
- **Test**: 15 000 samples for evaluation only (`samples_te.pt`).

This ensures an unbiased performance measure on unseen joint configurations.
"""

import os, sys
from typing import List, Optional
from time import time

from ikflow.config import DATASET_DIR, ALL_DATASET_TAGS, DATASET_TAG_NON_SELF_COLLIDING
from ikflow.utils import (
    safe_mkdir,
    print_tensor_stats,
    get_sum_joint_limit_range,
    get_dataset_filepaths,
    assert_joint_angle_tensor_in_joint_limits,
)
from jrl.robots import Robot
import torch

# === USER CONFIGURATION ===
# Hardcode your parameters here so you can run without CLI arguments
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
print(f"Base directory: {base_dir}")
sys.path.append(base_dir)
URDF_PATH = os.path.join(base_dir, "ur5e_utils_mujoco/ur5e.urdf")
#URDF_PATH = r"C:\Users\chris\Desktop\Test\ur5e.urdf"
TRAINING_SET_SIZE = 2_000_000
TEST_SET_SIZE = 15_000
ONLY_NON_SELF_COLLIDING = True
# ===========================


def print_saved_datasets_stats(tags: List[str], robots: Optional[List[Robot]] = []):
    """Print summary statistics for each dataset. Optionally print out joint limits of robots."""
    print("\tSummary info on all saved datasets:")

    def print_joint_limits(limits, robot_name=""):
        print(f"print_joint_limits('{robot_name}')")
        sum_range = 0
        for idx, (l, u) in enumerate(limits):
            sum_range += u - l
            print(f"  joint_{idx}: ({l},\t{u})")
        print(f"  sum_range: {sum_range}")

    for r in robots:
        print_joint_limits(r.actuated_joints_limits, robot_name=r.name)

    sp = "\t"
    print(f"\nrobot {sp} dataset_name {sp} sum_joint_range")
    print(f"----- {sp} ------------ {sp} ---------------")

    for dataset_directory, dirs, files in os.walk(DATASET_DIR):
        if len(dirs) > 0:
            continue
        dataset_name = os.path.basename(dataset_directory)
        robot_name = dataset_name.split("_")[0]
        try:
            samples_tr_fp, _, _, _, _ = get_dataset_filepaths(dataset_directory, tags)
            samples_tr = torch.load(samples_tr_fp)
            sum_joint_range = get_sum_joint_limit_range(samples_tr)
            print(f"{robot_name} {sp} {dataset_name} {sp} {sum_joint_range}")
        except FileNotFoundError:
            samples_tr_fp, _, _, _, _ = get_dataset_filepaths(dataset_directory, ALL_DATASET_TAGS)
            samples_tr = torch.load(samples_tr_fp)
            sum_joint_range = get_sum_joint_limit_range(samples_tr)
            print(f"{robot_name} {sp} {dataset_name} {sp} {sum_joint_range}")


def save_dataset_to_disk(
    robot: Robot,
    dataset_directory: str,
    training_set_size: int,
    test_set_size: int,
    only_non_self_colliding: bool,
    tags: List[str],
    joint_limit_eps: float = 1e-6,
):
    """
    Save training & test set to the provided directory.
    """
    safe_mkdir(dataset_directory)
    if only_non_self_colliding:
        assert DATASET_TAG_NON_SELF_COLLIDING in tags, (
            f"Error: If `only_non_self_colliding` is True, "
            f"'{DATASET_TAG_NON_SELF_COLLIDING}' must be in `tags`."
        )

    # Build test and training sets
    samples_te, poses_te = robot.sample_joint_angles_and_poses(
        test_set_size,
        joint_limit_eps=joint_limit_eps,
        only_non_self_colliding=only_non_self_colliding,
        tqdm_enabled=True,
    )
    samples_tr, poses_tr = robot.sample_joint_angles_and_poses(
        training_set_size,
        joint_limit_eps=joint_limit_eps,
        only_non_self_colliding=only_non_self_colliding,
        tqdm_enabled=True,
    )

    # Filepaths
    (
        samples_tr_file_path,
        poses_tr_file_path,
        samples_te_file_path,
        poses_te_file_path,
        info_filepath,
    ) = get_dataset_filepaths(dataset_directory, tags)

    # Convert to tensors
    samples_tr = torch.tensor(samples_tr, dtype=torch.float32)
    poses_tr = torch.tensor(poses_tr, dtype=torch.float32)
    samples_te = torch.tensor(samples_te, dtype=torch.float32)
    poses_te = torch.tensor(poses_te, dtype=torch.float32)

    # Sanity checks
    for arr in [samples_tr, samples_te, poses_tr, poses_te]:
        for i in range(arr.shape[1]):
            assert torch.std(arr[:, i]) > 0.001, f"Error: Column {i} has zero stdev"
    assert_joint_angle_tensor_in_joint_limits(robot.actuated_joints_limits, samples_tr, "samples_tr", 0.0)
    assert_joint_angle_tensor_in_joint_limits(robot.actuated_joints_limits, samples_te, "samples_te", 0.0)

    # Write info file
    with open(info_filepath, "w") as f:
        f.write("Dataset info\n")
        f.write(f"  robot:             {robot.name}\n")
        f.write(f"  dataset_directory: {dataset_directory}\n")
        f.write(f"  training_set_size: {training_set_size}\n")
        f.write(f"  test_set_size:     {test_set_size}\n\n")
        print_tensor_stats(samples_tr, writable=f, name="samples_tr")
        print_tensor_stats(poses_tr, writable=f, name="poses_tr")
        print_tensor_stats(samples_te, writable=f, name="samples_te")
        print_tensor_stats(poses_te, writable=f, name="poses_te")

    # Save tensors
    torch.save(samples_tr, samples_tr_file_path)
    torch.save(poses_tr, poses_tr_file_path)
    torch.save(samples_te, samples_te_file_path)
    torch.save(poses_te, poses_te_file_path)


# Helper to build tag list without argparse
def _get_tags(only_non_self_colliding: bool) -> List[str]:
    tags = []
    if only_non_self_colliding:
        tags.append(DATASET_TAG_NON_SELF_COLLIDING)
    else:
        print("====" * 10)
        print("WARNING: Saving dataset with self-colliding configurations. This is not recommended.")
    tags.sort()
    for tag in tags:
        assert tag in ALL_DATASET_TAGS
    return tags

# === MAIN EXECUTION ===
if __name__ == "__main__":
    # Prepare robot
    robot = Robot(
        name="ur5e_custom",
        urdf_filepath=URDF_PATH,
        active_joints=[
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ],
        base_link="base_link",
        end_effector_link_name="wrist_3_link",
        ignored_collision_pairs=[],
        collision_capsules_by_link=None,
    )
    tags = _get_tags(ONLY_NON_SELF_COLLIDING)

    # Build and save dataset
    print(f"Building dataset for robot: {robot.name}")
    suffix = f"_{'_'.join(tags)}" if tags else ""
    dset_directory = os.path.join(DATASET_DIR, f"{robot.name}{suffix}")
    t0 = time()
    save_dataset_to_disk(
        robot,
        dset_directory,
        TRAINING_SET_SIZE,
        TEST_SET_SIZE,
        ONLY_NON_SELF_COLLIDING,
        tags,
        joint_limit_eps=0.004363323129985824,  # ~0.25°
    )
    print(f"Saved dataset with {TRAINING_SET_SIZE} samples in {time() - t0:.2f} seconds")
    print_saved_datasets_stats(tags)
