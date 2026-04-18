from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import re

import numpy as np

from _bootstrap import add_project_root_to_path

add_project_root_to_path()

from simulation.config import ObjectSpec, default_config
from simulation.trial_runner import run_grasp_trial


def _offset_targets(base: tuple[tuple[str, float], ...], updates: dict[str, float]) -> tuple[tuple[str, float], ...]:
    out: list[tuple[str, float]] = []
    for name, value in base:
        out.append((name, value + updates.get(name, 0.0)))
    return tuple(out)


def _score_trial(row: dict[str, float | int | str]) -> float:
    success = float(row["success"])
    contact = float(row["contact_count_obj_gripper_peak"])
    z_delta = float(row["object_height_delta"])
    tactile = float(row["tactile_contact_mean"])
    slips = float(row["slip_event_count"])

    # Maximize lift success and object-gripper contact while penalizing high slip.
    return 6.0 * success + 0.9 * contact + 70.0 * z_delta + 0.5 * tactile - 0.15 * slips


def _sweep_candidates(
    max_combos: int,
) -> list[tuple[dict[str, float], tuple[float, float, float]]]:
    joint_offsets = {
        "hindarm_ctrl": [-0.08, -0.04, 0.0, 0.04, 0.08],
        "forearm_ctrl": [-0.10, -0.05, 0.0, 0.05, 0.10],
        "wrist_ctrl": [-0.08, -0.04, 0.0, 0.04, 0.08],
        "hand_ctrl": [-0.06, -0.03, 0.0, 0.03, 0.06],
    }

    spawn_offsets = [-0.006, -0.003, 0.0, 0.003, 0.006]

    combos: list[tuple[dict[str, float], tuple[float, float, float]]] = []
    for h in joint_offsets["hindarm_ctrl"]:
        for f in joint_offsets["forearm_ctrl"]:
            for w in joint_offsets["wrist_ctrl"]:
                for hd in joint_offsets["hand_ctrl"]:
                    for ox in spawn_offsets:
                        for oy in spawn_offsets:
                            for oz in spawn_offsets:
                                combos.append(
                                    (
                                        {
                                            "hindarm_ctrl": h,
                                            "forearm_ctrl": f,
                                            "wrist_ctrl": w,
                                            "hand_ctrl": hd,
                                        },
                                        (ox, oy, oz),
                                    )
                                )

    if len(combos) <= max_combos:
        return combos

    rng = np.random.default_rng(42)
    idx = rng.choice(np.arange(len(combos)), size=max_combos, replace=False)
    return [combos[int(i)] for i in idx]


def _update_config_file(config_path: Path, arm_targets: tuple[tuple[str, float], ...], nominal_spawn: tuple[float, float, float]) -> None:
    text = config_path.read_text(encoding="utf-8")

    arm_lines = ["    arm_grasp_targets: tuple[tuple[str, float], ...] = ("]
    for name, value in arm_targets:
        arm_lines.append(f'        ("{name}", {value:.3f}),')
    arm_lines.append("    )")
    arm_block = "\n".join(arm_lines)

    spawn_line = (
        "    spawn_nominal_xyz: tuple[float, float, float] = "
        f"({nominal_spawn[0]:.3f}, {nominal_spawn[1]:.3f}, {nominal_spawn[2]:.3f})"
    )

    text = re.sub(
        r"\s*arm_grasp_targets: tuple\[tuple\[str, float\], \.\.\.\] = \(\n(?:\s*\([^\n]*\),\n)+\s*\)",
        "\n" + arm_block,
        text,
        count=1,
    )
    text = re.sub(
        r"\s*spawn_nominal_xyz: tuple\[float, float, float\] = \([^\)]*\)",
        "\n" + spawn_line,
        text,
        count=1,
    )

    config_path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Automatic grasp pose calibration sweep")
    parser.add_argument("--max-combos", type=int, default=120, help="Maximum sampled combinations from the full sweep")
    parser.add_argument("--repeats", type=int, default=2, help="Repeats per candidate")
    parser.add_argument("--write-config", action="store_true", help="Write best arm_grasp_targets and spawn_nominal_xyz back to simulation/config.py")
    args = parser.parse_args()

    base_cfg = default_config()
    candidates = _sweep_candidates(args.max_combos)

    base_spawn = base_cfg.spawn_nominal_xyz
    force_candidates = [4.0, 7.0, 10.0]

    best_score = -1e18
    best_arm = base_cfg.arm_grasp_targets
    best_spawn = base_spawn

    trial_counter = 0
    scenario_id = 0

    for joint_updates, spawn_offset in candidates:
        candidate_arm = _offset_targets(base_cfg.arm_grasp_targets, joint_updates)
        spawn = (
            base_spawn[0] + spawn_offset[0],
            base_spawn[1] + spawn_offset[1],
            base_spawn[2] + spawn_offset[2],
        )

        cfg = replace(base_cfg, arm_grasp_targets=candidate_arm)

        candidate_scores: list[float] = []
        for _ in range(args.repeats):
            for force_cmd in force_candidates:
                trial_counter += 1
                obj = ObjectSpec(
                    object_type="box",
                    size_xyz=(0.006, 0.006, 0.006),
                    mass=0.08,
                    spawn_pos_xyz=spawn,
                )
                rec = run_grasp_trial(cfg, scenario_id=scenario_id, trial_id=trial_counter, obj=obj, grip_command=force_cmd)
                row = rec.to_row()
                candidate_scores.append(_score_trial(row))

            scenario_id += 1

        avg_score = float(np.mean(candidate_scores)) if candidate_scores else -1e18
        if avg_score > best_score:
            best_score = avg_score
            best_arm = candidate_arm
            best_spawn = spawn

    print("Calibration completed.")
    print(f"Best score: {best_score:.4f}")
    print(f"Best spawn nominal xyz: {best_spawn}")
    print("Best arm_grasp_targets:")
    for name, value in best_arm:
        print(f"  {name}: {value:.4f}")

    if args.write_config:
        config_path = Path(__file__).resolve().parents[1] / "simulation" / "config.py"
        _update_config_file(config_path, best_arm, best_spawn)
        print(f"Updated tuned defaults in: {config_path}")


if __name__ == "__main__":
    main()
