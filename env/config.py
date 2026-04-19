from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EnvConfig:
    model_path: Path
    max_steps: int = 500
    target_cube_pos: tuple[float, float, float] = (0.22, 0.0, 0.08)
    lift_success_z: float = 0.12
    ee_body_candidates: tuple[str, ...] = ("gripper_Gripper", "hand_Hand", "Gripper", "Hand")
    actuator_names: tuple[str, ...] = (
        "hip_ctrl",
        "hindarm_ctrl",
        "forearm_ctrl",
        "wrist_ctrl",
        "hand_ctrl",
        "gripper_ctrl",
    )


def default_env_config() -> EnvConfig:
    root = Path(__file__).resolve().parents[1]
    return EnvConfig(model_path=root / "assets" / "main.xml")
