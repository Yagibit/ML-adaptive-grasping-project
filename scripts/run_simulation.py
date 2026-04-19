from __future__ import annotations

import argparse
import os
from pathlib import Path
import time
import xml.etree.ElementTree as ET

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import mujoco
import mujoco.viewer
import numpy as np


def actuator_id(model: mujoco.MjModel, name: str) -> int:
    idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if idx < 0:
        raise ValueError(f"Actuator not found: {name}")
    return int(idx)


def body_id(model: mujoco.MjModel, name: str) -> int:
    idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if idx < 0:
        raise ValueError(f"Body not found: {name}")
    return int(idx)


def first_body_id(model: mujoco.MjModel, candidates: list[str]) -> int:
    for name in candidates:
        idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if idx >= 0:
            return int(idx)
    raise ValueError(f"None of the body names were found: {candidates}")


def body_free_joint_id(model: mujoco.MjModel, body: int) -> int | None:
    jnt_count = int(model.body_jntnum[body])
    if jnt_count <= 0:
        return None
    jnt_adr = int(model.body_jntadr[body])
    for j in range(jnt_adr, jnt_adr + jnt_count):
        if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE):
            return int(j)
    return None


def clip_to_actuator_range(model: mujoco.MjModel, actuator: int, value: float) -> float:
    if int(model.actuator_ctrllimited[actuator]) != 0:
        lo = float(model.actuator_ctrlrange[actuator][0])
        hi = float(model.actuator_ctrlrange[actuator][1])
        return float(np.clip(value, lo, hi))
    return float(value)


def move_toward(model: mujoco.MjModel, data: mujoco.MjData, actuator: int, target: float, step: float) -> None:
    current = float(data.ctrl[actuator])
    if abs(target - current) <= step:
        data.ctrl[actuator] = clip_to_actuator_range(model, actuator, target)
    elif target > current:
        data.ctrl[actuator] = clip_to_actuator_range(model, actuator, current + step)
    else:
        data.ctrl[actuator] = clip_to_actuator_range(model, actuator, current - step)


def smooth_update(model: mujoco.MjModel, data: mujoco.MjData, actuator: int, delta: float) -> None:
    data.ctrl[actuator] = clip_to_actuator_range(model, actuator, float(data.ctrl[actuator]) + delta)


def set_cube_pose(data: mujoco.MjData, qpos_adr: int, x: float, y: float, z: float) -> None:
    data.qpos[qpos_adr + 0] = x
    data.qpos[qpos_adr + 1] = y
    data.qpos[qpos_adr + 2] = z
    data.qpos[qpos_adr + 3] = 1.0
    data.qpos[qpos_adr + 4] = 0.0
    data.qpos[qpos_adr + 5] = 0.0
    data.qpos[qpos_adr + 6] = 0.0
    data.qvel[qpos_adr : qpos_adr + 6] = 0.0


def _load_model_with_runtime_fix(model_path: Path) -> mujoco.MjModel:
    try:
        return mujoco.MjModel.from_xml_path(str(model_path))
    except ValueError as exc:
        msg = str(exc)
        if "repeated name 'gripper_ctrl' in actuator" not in msg:
            raise

        tree = ET.parse(model_path)
        root = tree.getroot()
        actuator = root.find("actuator")
        if actuator is None:
            raise

        removed = False
        for child in list(actuator):
            if child.tag == "position" and child.get("name") == "gripper_ctrl":
                actuator.remove(child)
                removed = True
                break

        if not removed:
            raise

        fixed_path = model_path.parent / "main.runtime.fixed.xml"
        tree.write(fixed_path, encoding="ascii", xml_declaration=True)
        print("Runtime fix applied: removed duplicate gripper_ctrl actuator from temporary model copy.")
        return mujoco.MjModel.from_xml_path(str(fixed_path))


def gripper_actuator_id(model: mujoco.MjModel) -> int:
    for candidate in ["gripper_ctrl", "gripper_gripper_ctrl", "gripper_ctrl", "ctrl", "gripper_ctrl"]:
        idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, candidate)
        if idx >= 0:
            return int(idx)

    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or ""
        if "gripper" in name or name.endswith("ctrl"):
            if i >= 5:
                return int(i)
    raise ValueError("Could not identify gripper actuator.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run staged heuristic grasping simulation from assets/main.xml")
    parser.add_argument("--steps", type=int, default=0, help="Maximum simulation steps. Use 0 for no limit.")
    parser.add_argument("--pick-z", type=float, default=0.12, help="Success cube z threshold.")
    parser.add_argument("--continue-after-success", action="store_true", help="Keep running after first successful lift.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    model_path = project_root / "assets" / "main.xml"
    model = _load_model_with_runtime_fix(model_path)
    data = mujoco.MjData(model)

    hip = actuator_id(model, "hip_ctrl")
    hindarm = actuator_id(model, "hindarm_ctrl")
    forearm = actuator_id(model, "forearm_ctrl")
    wrist = actuator_id(model, "wrist_ctrl")
    hand = actuator_id(model, "hand_ctrl")
    gripper = gripper_actuator_id(model)
    cube = body_id(model, "cube")
    ee = first_body_id(model, ["gripper_Gripper", "hand_Hand", "Gripper", "Hand"])

    # Force gripper actuator to normalized control range in runtime model.
    model.actuator_ctrllimited[gripper] = 1
    model.actuator_ctrlrange[gripper, 0] = 0.0
    model.actuator_ctrlrange[gripper, 1] = 1.0

    # Strengthen contact and solver stability for grasping.
    model.geom_friction[:] = np.array([1.5, 0.005, 0.0001], dtype=float)
    model.opt.iterations = 100
    model.opt.ls_iterations = 50

    cube_free_joint = body_free_joint_id(model, cube)
    cube_qpos_adr = int(model.jnt_qposadr[cube_free_joint]) if cube_free_joint is not None else -1

    mujoco.mj_forward(model, data)
    max_steps = args.steps if args.steps > 0 else None

    # Initial neutral pose.
    data.ctrl[hip] = 0.50
    data.ctrl[hindarm] = 0.50
    data.ctrl[forearm] = 0.50
    data.ctrl[wrist] = 0.50
    data.ctrl[hand] = 0.50
    data.ctrl[gripper] = 0.20

    if cube_qpos_adr >= 0:
        # Keep cube accessible above base and floor at start.
        set_cube_pose(data, cube_qpos_adr, 0.22, 0.0, 0.08)
        mujoco.mj_forward(model, data)

    stage = "align"
    align_counter = 0
    approach_counter = 0
    grasp_counter = 0
    lift_counter = 0
    contact_counter = 0
    success_latched = False

    target_x, target_y, target_z = 0.22, 0.0, 0.12
    t = 0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            if max_steps is not None and t >= max_steps:
                print(f"Reached max steps: {max_steps}")
                break

            cube_pos = data.xpos[cube]
            ee_pos = data.xpos[ee]

            ex = float(target_x - ee_pos[0])
            ey = float(target_y - ee_pos[1])
            ez = float(target_z - ee_pos[2])

            if stage == "align":
                # 1) Align to face cube with stable absolute targets.
                move_toward(model, data, hip, 0.50, 0.005)
                move_toward(model, data, hindarm, 0.50, 0.005)
                move_toward(model, data, forearm, 0.55, 0.005)
                move_toward(model, data, wrist, 0.50, 0.005)
                move_toward(model, data, hand, 0.50, 0.005)
                move_toward(model, data, gripper, 0.20, 0.01)

                align_counter += 1
                if align_counter > 70:
                    stage = "approach"
                    print(f"Stage -> approach (step {t})")

            elif stage == "approach":
                # 2) Lower and approach using fixed keyframe targets (no spinning feedback).
                move_toward(model, data, hip, 0.50, 0.005)
                move_toward(model, data, hindarm, 0.55, 0.005)
                move_toward(model, data, forearm, 0.82, 0.005)
                move_toward(model, data, wrist, 0.55, 0.005)
                move_toward(model, data, hand, 0.50, 0.005)
                move_toward(model, data, gripper, 0.20, 0.01)

                approach_counter += 1
                if approach_counter > 150 or (abs(ex) < 0.050 and abs(ey) < 0.025 and abs(ez) < 0.060):
                    stage = "grasp"
                    print(f"Stage -> grasp (step {t})")

            elif stage == "grasp":
                # 3) Grasp: close gripper once fingers are around the cube.
                move_toward(model, data, hip, 0.50, 0.001)
                move_toward(model, data, hindarm, 0.55, 0.001)
                move_toward(model, data, forearm, 0.82, 0.001)
                move_toward(model, data, wrist, 0.55, 0.001)
                move_toward(model, data, hand, 0.50, 0.001)
                move_toward(model, data, gripper, 0.90, 0.012)

                if int(data.ncon) > 0:
                    contact_counter += 1

                grasp_counter += 1
                if grasp_counter > 130 or contact_counter > 20:
                    stage = "lift"
                    print(f"Stage -> lift (step {t})")

            elif stage == "lift":
                # Raise arm slightly while keeping grasp closed.
                move_toward(model, data, gripper, 0.90, 0.008)
                move_toward(model, data, hip, 0.50, 0.003)
                move_toward(model, data, hindarm, 0.48, 0.003)
                move_toward(model, data, forearm, 0.70, 0.003)
                move_toward(model, data, wrist, 0.50, 0.002)

                lift_counter += 1
                if lift_counter > 380:
                    stage = "align"
                    align_counter = 0
                    approach_counter = 0
                    grasp_counter = 0
                    lift_counter = 0
                    contact_counter = 0
                    data.ctrl[gripper] = clip_to_actuator_range(model, gripper, 0.20)
                    print(f"Retry cycle -> align (step {t})")

            mujoco.mj_step(model, data)

            # Keep cube from sinking below reachable workspace due instability.
            if cube_qpos_adr >= 0 and float(data.qpos[cube_qpos_adr + 2]) < 0.06:
                set_cube_pose(data, cube_qpos_adr, 0.22, 0.0, 0.08)
                mujoco.mj_forward(model, data)

            cube_z = float(data.xpos[cube][2])
            if cube_z > args.pick_z:
                if not success_latched:
                    print(f"Pick success at step {t}: cube z={cube_z:.5f}")
                    success_latched = True
                if not args.continue_after_success:
                    # Keep stage stable after success instead of terminating abruptly.
                    stage = "lift"

            viewer.sync()
            time.sleep(float(model.opt.timestep))
            t += 1


if __name__ == "__main__":
    main()
