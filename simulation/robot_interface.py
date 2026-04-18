from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass
class ControlIndices:
    arm_actuators: dict[str, int]
    gripper_actuator: int
    gripper_joint_ids: list[int]
    trial_object_body_id: int
    trial_object_geom_id: int
    gripper_geom_ids: list[int]


def _name_or_empty(name: str | None) -> str:
    return name or ""


def discover_indices(model: mujoco.MjModel) -> ControlIndices:
    arm_actuators: dict[str, int] = {}
    gripper_actuator = -1

    for i in range(model.nu):
        act_name = _name_or_empty(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i))
        if any(k in act_name for k in ("hip_ctrl", "hindarm_ctrl", "forearm_ctrl", "wrist_ctrl", "hand_ctrl")):
            arm_actuators[act_name] = i

        if "gripper" in act_name and "ctrl" in act_name:
            gripper_actuator = i

    if gripper_actuator < 0:
        for i in range(model.nu):
            joint_id = int(model.actuator_trnid[i, 0])
            jname = _name_or_empty(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id))
            if "servo_gear" in jname:
                gripper_actuator = i
                break

    gripper_joint_ids: list[int] = []
    for j in range(model.njnt):
        jname = _name_or_empty(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j))
        if "gripper" in jname or "servo_gear" in jname or "idol_gear" in jname:
            gripper_joint_ids.append(j)

    trial_object_body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trial_object"))
    trial_object_geom_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "trial_object_geom"))

    gripper_geom_ids: list[int] = []
    for g in range(model.ngeom):
        gname = _name_or_empty(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g))
        if "gripper_" in gname and "collision" in gname:
            gripper_geom_ids.append(g)

    if gripper_actuator < 0:
        raise RuntimeError("Unable to discover gripper actuator. Inspect actuator names with scripts/inspect_model.py.")

    return ControlIndices(
        arm_actuators=arm_actuators,
        gripper_actuator=gripper_actuator,
        gripper_joint_ids=gripper_joint_ids,
        trial_object_body_id=trial_object_body_id,
        trial_object_geom_id=trial_object_geom_id,
        gripper_geom_ids=gripper_geom_ids,
    )


def set_arm_targets(model: mujoco.MjModel, data: mujoco.MjData, arm_targets: tuple[tuple[str, float], ...], arm_actuator_map: dict[str, int]) -> None:
    for actuator_name, target in arm_targets:
        actuator_id = arm_actuator_map.get(actuator_name)
        if actuator_id is not None:
            data.ctrl[actuator_id] = target


def step_n(model: mujoco.MjModel, data: mujoco.MjData, steps: int) -> None:
    for _ in range(steps):
        mujoco.mj_step(model, data)


def gripper_joint_position_mean(model: mujoco.MjModel, data: mujoco.MjData, gripper_joint_ids: list[int]) -> float:
    if not gripper_joint_ids:
        return 0.0

    values: list[float] = []
    for joint_id in gripper_joint_ids:
        qpos_addr = int(model.jnt_qposadr[joint_id])
        values.append(float(data.qpos[qpos_addr]))
    return float(np.mean(values))
