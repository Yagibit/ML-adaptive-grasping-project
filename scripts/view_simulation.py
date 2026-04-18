from __future__ import annotations

import argparse

import mujoco
import mujoco.viewer
import numpy as np

from _bootstrap import add_project_root_to_path

add_project_root_to_path()

from simulation.config import ObjectSpec, default_config  # noqa: E402
from simulation.scene_builder import build_scene_xml  # noqa: E402
from simulation.data_extraction import extract_contact_stats  # noqa: E402
from simulation.robot_interface import discover_indices, set_arm_targets  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive MuJoCo viewer for the existing robot and dynamic trial object")
    parser.add_argument("--object-type", choices=["box", "cylinder"], default="box")
    parser.add_argument("--grip-command", type=float, default=6.0)
    parser.add_argument("--mass", type=float, default=0.08)
    parser.add_argument("--size", type=float, default=0.006)
    parser.add_argument("--height", type=float, default=0.006)
    args = parser.parse_args()

    cfg = default_config()
    obj = ObjectSpec(
        object_type=args.object_type,
        size_xyz=(args.size, args.size, args.height),
        mass=args.mass,
        spawn_pos_xyz=(-0.040, 0.0, 0.020),
    )

    scene_path = build_scene_xml(cfg.base_model_path, cfg.generated_scene_dir / "interactive_view.xml", obj)
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    ids = discover_indices(model)

    mujoco.mj_resetData(model, data)
    set_arm_targets(model, data, cfg.arm_pregrasp_targets, ids.arm_actuators)
    data.ctrl[ids.gripper_actuator] = cfg.gripper_open_command
    mujoco.mj_forward(model, data)
    adaptive_cmd = float(args.grip_command)

    print("Interactive commands:")
    print("  open   - open gripper")
    print("  close  - close gripper")
    print("  grip X - set gripper command to X")
    print("  arm    - reapply pre-grasp arm targets")
    print("  approach - move lower arm to grasp pose")
    print("  lift   - move lower arm to lift pose")
    print("  adapt on|off - toggle tactile adaptive grip")
    print("  stats  - print contact/force/object-height feedback")
    print("  reset  - reset simulation")
    print("  quit   - exit viewer")

    adaptive_on = False

    def smooth_move(targets: tuple[tuple[str, float], ...], steps: int = 180) -> None:
        for _ in range(max(1, steps)):
            set_arm_targets(model, data, targets, ids.arm_actuators)
            mujoco.mj_step(model, data)
            if not viewer.is_running():
                return
            viewer.sync()

    def print_stats() -> None:
        stats = extract_contact_stats(model, data, ids.trial_object_geom_id, ids.gripper_geom_ids)
        obj_z = float(data.xpos[ids.trial_object_body_id][2])
        print(
            f"object_z={obj_z:.6f}, obj_gripper_contacts={stats['contact_count_obj_gripper']:.1f}, "
            f"normal_force={stats['mean_contact_normal_force']:.4f}"
        )

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            cmd = input("viewer> ").strip().lower()
            if cmd in {"quit", "exit", "q"}:
                break
            if cmd == "open":
                data.ctrl[ids.gripper_actuator] = cfg.gripper_open_command
            elif cmd == "close":
                data.ctrl[ids.gripper_actuator] = args.grip_command
            elif cmd.startswith("grip "):
                try:
                    value = float(cmd.split(maxsplit=1)[1])
                except ValueError:
                    print("Invalid grip value.")
                    continue
                data.ctrl[ids.gripper_actuator] = value
                adaptive_cmd = value
            elif cmd == "arm":
                set_arm_targets(model, data, cfg.arm_pregrasp_targets, ids.arm_actuators)
            elif cmd == "approach":
                smooth_move(cfg.arm_grasp_targets)
                continue
            elif cmd == "lift":
                smooth_move(cfg.arm_lift_targets)
                continue
            elif cmd.startswith("adapt "):
                mode = cmd.split(maxsplit=1)[1].strip()
                if mode == "on":
                    adaptive_on = True
                    print("Adaptive grip: ON")
                elif mode == "off":
                    adaptive_on = False
                    print("Adaptive grip: OFF")
                else:
                    print("Use: adapt on|off")
                continue
            elif cmd == "stats":
                print_stats()
                continue
            elif cmd == "reset":
                mujoco.mj_resetData(model, data)
                set_arm_targets(model, data, cfg.arm_pregrasp_targets, ids.arm_actuators)
                data.ctrl[ids.gripper_actuator] = cfg.gripper_open_command
                mujoco.mj_forward(model, data)
                adaptive_cmd = float(args.grip_command)
            else:
                print("Unknown command.")
                continue

            for _ in range(20):
                mujoco.mj_step(model, data)
                if adaptive_on:
                    stats = extract_contact_stats(model, data, ids.trial_object_geom_id, ids.gripper_geom_ids)
                    if stats["contact_count_obj_gripper"] < cfg.tactile_contact_target:
                        adaptive_cmd += cfg.adaptive_gain_contact
                    if stats["mean_contact_normal_force"] < cfg.tactile_force_target_n:
                        adaptive_cmd += cfg.adaptive_gain_force
                    elif stats["mean_contact_normal_force"] > cfg.tactile_force_high_n:
                        adaptive_cmd -= cfg.adaptive_gain_force
                    adaptive_cmd = float(np.clip(adaptive_cmd, cfg.gripper_min_command, cfg.gripper_max_command))
                    data.ctrl[ids.gripper_actuator] = adaptive_cmd
                if not viewer.is_running():
                    return
                viewer.sync()


if __name__ == "__main__":
    main()
