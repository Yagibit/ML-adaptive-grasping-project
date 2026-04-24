from __future__ import annotations

import argparse
import os
from pathlib import Path
import time

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
    return mujoco.MjModel.from_xml_path(str(model_path))


def gripper_actuator_ids(model: mujoco.MjModel) -> tuple[int, int]:
    left = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper_left_slide_ctrl")
    right = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper_right_slide_ctrl")
    if left < 0 or right < 0:
        raise ValueError("Could not identify gripper actuators.")
    return int(left), int(right)


def set_gripper_command(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    actuator_ids: tuple[int, int],
    command: float,
) -> None:
    # External command space is binary-like: >=0 opens, <0 closes.
    # Both slide joints receive the same target so jaw motion stays symmetric.
    open_target = 0.0
    close_target = -0.02
    target = open_target if command >= 0.0 else close_target
    for actuator in actuator_ids:
        data.ctrl[actuator] = clip_to_actuator_range(model, actuator, target)


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
    gripper_left, gripper_right = gripper_actuator_ids(model)
    cube = body_id(model, "cube")
    ee = first_body_id(model, ["gripper_Gripper", "hand_Hand", "Gripper", "Hand"])

    # Apply global friction/solver overrides to reduce contact jitter in demo mode.
    model.geom_friction[:] = np.array([1.0, 0.5, 0.5], dtype=float)
    model.opt.iterations = 100
    model.opt.ls_iterations = 50

    cube_free_joint = body_free_joint_id(model, cube)
    cube_qpos_adr = int(model.jnt_qposadr[cube_free_joint]) if cube_free_joint is not None else -1

    mujoco.mj_forward(model, data)
    max_steps = args.steps if args.steps > 0 else None

    # Initial hard-coded pose for a stable presentation grasp.
    data.ctrl[hip] = 0.65
    data.ctrl[hindarm] = 0.70
    data.ctrl[forearm] = 0.40
    data.ctrl[wrist] = 0.25
    data.ctrl[hand] = 0.50
    set_gripper_command(model, data, (gripper_left, gripper_right), 0.8)

    if cube_qpos_adr >= 0:
        # Keep cube accessible above base and floor at start.
        set_cube_pose(data, cube_qpos_adr, 0.22, 0.0, 0.08)
        mujoco.mj_forward(model, data)

    stage = "approach"
    stage_counter = 0
    success_latched = False

    t = 0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            if max_steps is not None and t >= max_steps:
                print(f"Reached max steps: {max_steps}")
                break

            if stage == "approach":
                # Hold approach pose long enough for arm dynamics to settle.
                data.ctrl[hip] = 0.65
                data.ctrl[hindarm] = 0.70
                data.ctrl[forearm] = 0.40
                data.ctrl[wrist] = 0.25
                data.ctrl[hand] = 0.50
                set_gripper_command(model, data, (gripper_left, gripper_right), 0.8)

                stage_counter += 1
                if stage_counter >= 120:
                    stage = "close"
                    stage_counter = 0
                    print(f"Stage -> close (step {t})")

            elif stage == "close":
                # Close both fingers after the approach window.
                data.ctrl[hip] = 0.65
                data.ctrl[hindarm] = 0.70
                data.ctrl[forearm] = 0.40
                data.ctrl[wrist] = 0.25
                data.ctrl[hand] = 0.50
                set_gripper_command(model, data, (gripper_left, gripper_right), -0.8)

                stage_counter += 1
                if stage_counter >= 120:
                    stage = "lift"
                    stage_counter = 0
                    print(f"Stage -> lift (step {t})")

            elif stage == "lift":
                # Execute a small lift while maintaining closed gripper command.
                data.ctrl[hip] = 0.60
                data.ctrl[hindarm] = 0.58
                data.ctrl[forearm] = 0.36
                data.ctrl[wrist] = 0.22
                data.ctrl[hand] = 0.50
                set_gripper_command(model, data, (gripper_left, gripper_right), -0.8)

                stage_counter += 1
                if stage_counter >= 1000:
                    stage = "hold"
                    stage_counter = 0
                    print(f"Stage -> hold (step {t})")

            elif stage == "hold":
                data.ctrl[hip] = 0.60
                data.ctrl[hindarm] = 0.58
                data.ctrl[forearm] = 0.36
                data.ctrl[wrist] = 0.22
                data.ctrl[hand] = 0.50
                set_gripper_command(model, data, (gripper_left, gripper_right), -0.8)

            mujoco.mj_step(model, data)

            # Recover cube when it tunnels below the table due numerical instability.
            if cube_qpos_adr >= 0 and float(data.qpos[cube_qpos_adr + 2]) < 0.04:
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
