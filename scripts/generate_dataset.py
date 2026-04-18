from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from _bootstrap import add_project_root_to_path

add_project_root_to_path()

from plot_tactile_feedback import save_tactile_feedback_plots
from simulation.config import ObjectSpec, SimulationConfig, default_config
from simulation.trial_runner import run_grasp_trial


def _sample_object(rng: np.random.Generator, cfg: SimulationConfig) -> ObjectSpec:
    obj_type = str(rng.choice(["box", "cylinder"]))

    if obj_type == "box":
        sx = float(rng.uniform(0.004, 0.010))
        sy = float(rng.uniform(0.004, 0.010))
        sz = float(rng.uniform(0.004, 0.012))
    else:
        radius = float(rng.uniform(0.003, 0.008))
        half_height = float(rng.uniform(0.004, 0.012))
        sx, sy, sz = radius, radius, half_height

    mass = float(rng.uniform(0.02, 0.15))
    sx_nom, sy_nom, sz_nom = cfg.spawn_nominal_xyz
    sx_rng, sy_rng, sz_rng = cfg.spawn_offset_range_xyz
    spawn = (
        float(rng.uniform(sx_nom - sx_rng, sx_nom + sx_rng)),
        float(rng.uniform(sy_nom - sy_rng, sy_nom + sy_rng)),
        float(rng.uniform(sz_nom - sz_rng, sz_nom + sz_rng)),
    )
    return ObjectSpec(object_type=obj_type, size_xyz=(sx, sy, sz), mass=mass, spawn_pos_xyz=spawn)


def _attach_optimal_force_label(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["optimal_grip_command"] = np.nan

    grouped = out.groupby("scenario_id")
    for sid, g in grouped:
        successful = g[g["success"] == 1]
        if successful.empty:
            continue
        optimal = float(successful["adaptive_final_grip_command"].min())
        out.loc[out["scenario_id"] == sid, "optimal_grip_command"] = optimal

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate grip dataset from MuJoCo grasp trials")
    parser.add_argument("--num-scenarios", type=int, default=40)
    parser.add_argument("--forces-per-scenario", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("data/raw/grip_dataset.csv"))
    parser.add_argument("--plot", action="store_true", help="Generate tactile feedback plots after dataset creation")
    parser.add_argument("--plots-dir", type=Path, default=Path("data/processed/plots"))
    args = parser.parse_args()

    cfg = default_config()
    rng = np.random.default_rng(args.seed)

    rows: list[dict[str, float | int | str]] = []
    trial_id = 0

    for scenario_id in range(args.num_scenarios):
        obj = _sample_object(rng, cfg)
        force_values = np.linspace(cfg.gripper_min_command, cfg.gripper_max_command, args.forces_per_scenario)

        for force_cmd in force_values:
            trial_id += 1
            rec = run_grasp_trial(
                cfg=cfg,
                scenario_id=scenario_id,
                trial_id=trial_id,
                obj=obj,
                grip_command=float(force_cmd),
            )
            rows.append(rec.to_row())

    df = pd.DataFrame(rows)
    df = _attach_optimal_force_label(df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    success_rate = float(df["success"].mean()) if not df.empty else 0.0
    print(f"Saved dataset to: {args.output}")
    print(f"Rows: {len(df)}")
    print(f"Success rate: {success_rate:.3f}")
    print(f"Rows with optimal_grip_command: {int(df['optimal_grip_command'].notna().sum())}")

    if args.plot:
        plot_paths = save_tactile_feedback_plots(df, args.plots_dir)
        print("Saved plots:")
        for p in plot_paths:
            print(f"  {p}")


if __name__ == "__main__":
    main()
