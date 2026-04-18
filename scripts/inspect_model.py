from __future__ import annotations

import argparse

import mujoco

from _bootstrap import add_project_root_to_path

add_project_root_to_path()

from simulation.config import default_config
from simulation.robot_interface import discover_indices


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect joints and actuators discovered from assets/main.xml")
    parser.parse_args()

    cfg = default_config()
    model = mujoco.MjModel.from_xml_path(str(cfg.base_model_path))
    ids = discover_indices(model)

    print("=== Actuators ===")
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        print(f"{i:3d}: {name}")

    print("\n=== Joints ===")
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        print(f"{j:3d}: {name}")

    print("\n=== Discovered Indices ===")
    print(f"Gripper actuator id: {ids.gripper_actuator}")
    print(f"Arm actuator map: {ids.arm_actuators}")
    print(f"Gripper joint ids: {ids.gripper_joint_ids}")


if __name__ == "__main__":
    main()
