from __future__ import annotations

import argparse
import csv
from pathlib import Path

import mujoco
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


def run_trial(model: mujoco.MjModel, rng: np.random.Generator, settle_steps: int, action_steps: int) -> dict[str, float | int]:
    data = mujoco.MjData(model)

    hip = actuator_id(model, "hip_ctrl")
    hindarm = actuator_id(model, "hindarm_ctrl")
    forearm = actuator_id(model, "forearm_ctrl")
    wrist = actuator_id(model, "wrist_ctrl")
    hand = actuator_id(model, "hand_ctrl")
    cube = body_id(model, "cube")

    mujoco.mj_resetData(model, data)
    data.ctrl[hip] = 0.45
    data.ctrl[hindarm] = 0.55
    data.ctrl[forearm] = 0.50
    data.ctrl[wrist] = 0.50
    data.ctrl[hand] = 0.50

    for _ in range(settle_steps):
        mujoco.mj_step(model, data)

    z0 = float(data.xpos[cube][2])

    wrist_val = float(rng.uniform(0.0, 1.0))
    hand_val = float(rng.uniform(0.0, 1.0))
    data.ctrl[wrist] = wrist_val
    data.ctrl[hand] = hand_val

    peak_contacts = int(data.ncon)
    for _ in range(action_steps):
        mujoco.mj_step(model, data)
        if int(data.ncon) > peak_contacts:
            peak_contacts = int(data.ncon)

    z1 = float(data.xpos[cube][2])
    success = 1 if z1 > z0 + 1e-4 else 0

    return {
        "wrist_ctrl": wrist_val,
        "hand_ctrl": hand_val,
        "contact_count": peak_contacts,
        "object_height": z1,
        "success": success,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate grip dataset with normalized controls from assets/main.xml")
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("data/grip_dataset.csv"))
    parser.add_argument("--settle-steps", type=int, default=120)
    parser.add_argument("--action-steps", type=int, default=220)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    model_path = project_root / "assets" / "main.xml"
    model = mujoco.MjModel.from_xml_path(str(model_path))
    rng = np.random.default_rng(args.seed)

    output_path = args.output if args.output.is_absolute() else project_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [run_trial(model, rng, args.settle_steps, args.action_steps) for _ in range(args.trials)]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["wrist_ctrl", "hand_ctrl", "contact_count", "object_height", "success"])
        writer.writeheader()
        writer.writerows(rows)

    success_rate = sum(int(r["success"]) for r in rows) / max(1, len(rows))
    print(f"Saved dataset: {output_path}")
    print(f"Trials: {len(rows)}")
    print(f"Success rate: {success_rate:.3f}")


if __name__ == "__main__":
    main()
