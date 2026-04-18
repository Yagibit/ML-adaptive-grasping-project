from __future__ import annotations

import argparse

from _bootstrap import add_project_root_to_path

add_project_root_to_path()

from simulation.config import ObjectSpec, default_config
from simulation.trial_runner import run_grasp_trial


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one grasp trial as a simulation sanity check")
    parser.add_argument("--object-type", choices=["box", "cylinder"], default="box")
    parser.add_argument("--grip-command", type=float, default=6.0)
    parser.add_argument("--mass", type=float, default=0.08)
    parser.add_argument("--size", type=float, default=0.006, help="Half-size for box, radius for cylinder")
    parser.add_argument("--height", type=float, default=0.006, help="Half-height for cylinder or z half-size for box")
    args = parser.parse_args()

    cfg = default_config()

    if args.object_type == "box":
        size_xyz = (args.size, args.size, args.height)
    else:
        size_xyz = (args.size, args.size, args.height)

    obj = ObjectSpec(
        object_type=args.object_type,
        size_xyz=size_xyz,
        mass=args.mass,
        spawn_pos_xyz=(-0.040, 0.0, 0.020),
    )

    rec = run_grasp_trial(cfg=cfg, scenario_id=0, trial_id=1, obj=obj, grip_command=args.grip_command)
    print("Single trial result:")
    for k, v in rec.to_row().items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
