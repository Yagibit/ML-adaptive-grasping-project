from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SimulationConfig:
    workspace_root: Path
    base_model_path: Path
    generated_scene_dir: Path
    default_seed: int = 42
    settle_steps: int = 250
    approach_steps: int = 220
    close_steps: int = 350
    lift_steps: int = 220
    hold_steps: int = 220
    gripper_open_command: float = -8.0
    gripper_min_command: float = -12.0
    gripper_max_command: float = 12.0
    lift_success_threshold_m: float = 0.007
    stable_disp_threshold_m: float = 0.010
    stable_contact_min_count: int = 2
    tactile_contact_target: int = 2
    tactile_force_target_n: float = 1.5
    tactile_force_high_n: float = 6.0
    adaptive_gain_contact: float = 0.06
    adaptive_gain_force: float = 0.05
    adaptive_gain_slip: float = 0.10
    spawn_nominal_xyz: tuple[float, float, float] = (-0.040, 0.0, 0.020)
    spawn_offset_range_xyz: tuple[float, float, float] = (0.010, 0.010, 0.010)
    arm_pregrasp_targets: tuple[tuple[str, float], ...] = (
        ("hip_ctrl", 0.0),
        ("hindarm_ctrl", 0.20),
        ("forearm_ctrl", -0.55),
        ("wrist_ctrl", 0.35),
        ("hand_ctrl", 0.30),
    )
    arm_grasp_targets: tuple[tuple[str, float], ...] = (
        ("hip_ctrl", 0.0),
        ("hindarm_ctrl", 0.34),
        ("forearm_ctrl", -0.82),
        ("wrist_ctrl", 0.48),
        ("hand_ctrl", 0.34),
    )
    arm_lift_targets: tuple[tuple[str, float], ...] = (
        ("hip_ctrl", 0.0),
        ("hindarm_ctrl", 0.22),
        ("forearm_ctrl", -0.58),
        ("wrist_ctrl", 0.35),
        ("hand_ctrl", 0.28),
    )


@dataclass(frozen=True)
class ObjectSpec:
    object_type: str
    size_xyz: tuple[float, float, float]
    mass: float
    spawn_pos_xyz: tuple[float, float, float]


def default_config() -> SimulationConfig:
    root = Path(__file__).resolve().parents[1]
    return SimulationConfig(
        workspace_root=root,
        base_model_path=root / "assets" / "main.xml",
        generated_scene_dir=root / "assets",
    )
