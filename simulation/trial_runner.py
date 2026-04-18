from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np

from simulation.config import ObjectSpec, SimulationConfig
from simulation.data_extraction import extract_contact_stats
from simulation.robot_interface import (
    discover_indices,
    gripper_joint_position_mean,
    set_arm_targets,
    step_n,
)
from simulation.scene_builder import build_scene_xml


@dataclass
class TrialRecord:
    scenario_id: int
    trial_id: int
    object_type: str
    object_size_x: float
    object_size_y: float
    object_size_z: float
    object_mass: float
    grip_command: float
    gripper_joint_mean: float
    object_initial_z: float
    object_max_z: float
    object_final_z: float
    object_height_delta: float
    contact_count_total_peak: float
    contact_count_obj_gripper_peak: float
    mean_contact_normal_force: float
    tactile_contact_mean: float
    tactile_force_mean: float
    adaptive_final_grip_command: float
    slip_event_count: int
    success: int

    def to_row(self) -> dict[str, float | int | str]:
        return asdict(self)


def _scene_path(cfg: SimulationConfig, scenario_id: int, trial_id: int) -> Path:
    return cfg.generated_scene_dir / f"scene_s{scenario_id:04d}_t{trial_id:04d}.xml"


def _deactivate_non_trial_free_bodies(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Move any pre-existing free body far from the workspace to avoid trial contamination."""
    for joint_id in range(model.njnt):
        if int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
            continue

        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or ""
        if jname == "trial_object_freejoint":
            continue

        qpos_adr = int(model.jnt_qposadr[joint_id])
        data.qpos[qpos_adr : qpos_adr + 3] = np.array([0.0, 0.0, -2.0], dtype=float)
        data.qpos[qpos_adr + 3 : qpos_adr + 7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)


def _targets_to_map(targets: tuple[tuple[str, float], ...]) -> dict[str, float]:
    return {name: value for name, value in targets}


def _interpolate_arm_targets(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    arm_actuator_map: dict[str, int],
    start_targets: tuple[tuple[str, float], ...],
    end_targets: tuple[tuple[str, float], ...],
    steps: int,
) -> None:
    start_map = _targets_to_map(start_targets)
    end_map = _targets_to_map(end_targets)

    for s in range(max(1, steps)):
        alpha = (s + 1) / max(1, steps)
        blended = []
        for act_name, start_value in start_map.items():
            end_value = end_map.get(act_name, start_value)
            value = (1.0 - alpha) * start_value + alpha * end_value
            blended.append((act_name, value))
        set_arm_targets(model, data, tuple(blended), arm_actuator_map)
        mujoco.mj_step(model, data)


def _update_adaptive_grip(cfg: SimulationConfig, current_cmd: float, contact_count: float, normal_force: float, slipping: bool) -> float:
    updated = current_cmd

    if contact_count < float(cfg.tactile_contact_target):
        updated += cfg.adaptive_gain_contact

    if normal_force < cfg.tactile_force_target_n:
        updated += cfg.adaptive_gain_force
    elif normal_force > cfg.tactile_force_high_n:
        updated -= cfg.adaptive_gain_force

    if slipping:
        updated += cfg.adaptive_gain_slip

    return float(np.clip(updated, cfg.gripper_min_command, cfg.gripper_max_command))


def run_grasp_trial(
    cfg: SimulationConfig,
    scenario_id: int,
    trial_id: int,
    obj: ObjectSpec,
    grip_command: float,
) -> TrialRecord:
    scene_path = build_scene_xml(cfg.base_model_path, _scene_path(cfg, scenario_id, trial_id), obj)
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)

    ids = discover_indices(model)

    mujoco.mj_resetData(model, data)
    _deactivate_non_trial_free_bodies(model, data)
    mujoco.mj_forward(model, data)
    set_arm_targets(model, data, cfg.arm_pregrasp_targets, ids.arm_actuators)
    data.ctrl[ids.gripper_actuator] = cfg.gripper_open_command
    step_n(model, data, cfg.settle_steps)

    initial_z = float(data.xpos[ids.trial_object_body_id][2])

    max_z = initial_z
    peak_total_contacts = 0.0
    peak_obj_gripper_contacts = 0.0
    mean_forces: list[float] = []
    tactile_contacts: list[float] = []
    slip_event_count = 0

    _interpolate_arm_targets(
        model,
        data,
        ids.arm_actuators,
        cfg.arm_pregrasp_targets,
        cfg.arm_grasp_targets,
        cfg.approach_steps,
    )

    adaptive_grip_command = float(np.clip(grip_command, cfg.gripper_min_command, cfg.gripper_max_command))
    data.ctrl[ids.gripper_actuator] = adaptive_grip_command
    prev_z = float(data.xpos[ids.trial_object_body_id][2])

    for _ in range(cfg.close_steps):
        mujoco.mj_step(model, data)
        z = float(data.xpos[ids.trial_object_body_id][2])
        if z > max_z:
            max_z = z

        stats = extract_contact_stats(model, data, ids.trial_object_geom_id, ids.gripper_geom_ids)
        peak_total_contacts = max(peak_total_contacts, stats["contact_count_total"])
        peak_obj_gripper_contacts = max(peak_obj_gripper_contacts, stats["contact_count_obj_gripper"])
        mean_forces.append(stats["mean_contact_normal_force"])
        tactile_contacts.append(stats["contact_count_obj_gripper"])

        slipping = (z + 1e-5) < prev_z and stats["contact_count_obj_gripper"] >= 1.0
        if slipping:
            slip_event_count += 1
        adaptive_grip_command = _update_adaptive_grip(
            cfg,
            adaptive_grip_command,
            stats["contact_count_obj_gripper"],
            stats["mean_contact_normal_force"],
            slipping,
        )
        data.ctrl[ids.gripper_actuator] = adaptive_grip_command
        prev_z = z

    _interpolate_arm_targets(
        model,
        data,
        ids.arm_actuators,
        cfg.arm_grasp_targets,
        cfg.arm_lift_targets,
        cfg.lift_steps,
    )

    stable_counter = 0
    for _ in range(cfg.hold_steps):
        mujoco.mj_step(model, data)
        z = float(data.xpos[ids.trial_object_body_id][2])
        if z > max_z:
            max_z = z

        stats = extract_contact_stats(model, data, ids.trial_object_geom_id, ids.gripper_geom_ids)
        peak_total_contacts = max(peak_total_contacts, stats["contact_count_total"])
        peak_obj_gripper_contacts = max(peak_obj_gripper_contacts, stats["contact_count_obj_gripper"])
        mean_forces.append(stats["mean_contact_normal_force"])
        tactile_contacts.append(stats["contact_count_obj_gripper"])

        slipping = (z + 1e-5) < prev_z and stats["contact_count_obj_gripper"] >= 1.0
        if slipping:
            slip_event_count += 1
        adaptive_grip_command = _update_adaptive_grip(
            cfg,
            adaptive_grip_command,
            stats["contact_count_obj_gripper"],
            stats["mean_contact_normal_force"],
            slipping,
        )
        data.ctrl[ids.gripper_actuator] = adaptive_grip_command
        prev_z = z

        if int(stats["contact_count_obj_gripper"]) >= cfg.stable_contact_min_count:
            stable_counter += 1

    final_z = float(data.xpos[ids.trial_object_body_id][2])
    z_delta = max_z - initial_z

    stable_ratio = stable_counter / max(1, cfg.hold_steps)
    displacement = abs(final_z - initial_z)

    success = int(
        (z_delta >= cfg.lift_success_threshold_m)
        or (stable_ratio > 0.35 and displacement <= cfg.stable_disp_threshold_m)
    )

    return TrialRecord(
        scenario_id=scenario_id,
        trial_id=trial_id,
        object_type=obj.object_type,
        object_size_x=obj.size_xyz[0],
        object_size_y=obj.size_xyz[1],
        object_size_z=obj.size_xyz[2],
        object_mass=obj.mass,
        grip_command=float(grip_command),
        gripper_joint_mean=gripper_joint_position_mean(model, data, ids.gripper_joint_ids),
        object_initial_z=initial_z,
        object_max_z=max_z,
        object_final_z=final_z,
        object_height_delta=z_delta,
        contact_count_total_peak=peak_total_contacts,
        contact_count_obj_gripper_peak=peak_obj_gripper_contacts,
        mean_contact_normal_force=float(np.mean(mean_forces)) if mean_forces else 0.0,
        tactile_contact_mean=float(np.mean(tactile_contacts)) if tactile_contacts else 0.0,
        tactile_force_mean=float(np.mean(mean_forces)) if mean_forces else 0.0,
        adaptive_final_grip_command=adaptive_grip_command,
        slip_event_count=slip_event_count,
        success=success,
    )
