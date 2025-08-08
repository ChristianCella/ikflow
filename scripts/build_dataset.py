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
URDF_PATH = os.path.join(base_dir, "ur5e_utils_mujoco/ur5e/ur5e.urdf")
os.chdir(os.path.dirname(URDF_PATH))  # <-- Add this line
#URDF_PATH = r"C:\Users\chris\Desktop\Test\ur5e.urdf"
TRAINING_SET_SIZE = 2_000_000
TEST_SET_SIZE = 15_000
ONLY_NON_SELF_COLLIDING = True
# ===========================

import re

def patch_urdf_mesh_paths(urdf_path: str) -> str:
    with open(urdf_path, "r") as f:
        urdf_str = f.read()

    urdf_dir = os.path.dirname(urdf_path)

    def replace_mesh_path(match):
        mesh_path = match.group(1)
        full_path = os.path.abspath(os.path.join(urdf_dir, mesh_path))
        return f'<mesh filename="{full_path}"'

    patched_urdf = re.sub(r'<mesh filename="([^"]+)"', replace_mesh_path, urdf_str)

    patched_path = os.path.join(urdf_dir, "patched_ur5e.urdf")
    with open(patched_path, "w") as f:
        f.write(patched_urdf)

    print(f"[INFO] Patched URDF written to: {patched_path}")
    return patched_path



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
    patched_urdf_path = patch_urdf_mesh_paths(URDF_PATH)
    robot = Robot(
        name="ur5e_custom",
        urdf_filepath=patched_urdf_path,
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
