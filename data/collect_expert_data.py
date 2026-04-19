from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    import h5py
except ImportError:  # optional dependency at runtime
    h5py = None

from env.grasp_env import GraspEnv
from env.expert_policy import ScriptedExpertPolicy


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect expert demonstrations from staged scripted controller")
    parser.add_argument("--episodes", type=int, default=120)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-npz", type=Path, default=Path("data/datasets/expert_trajectories.npz"))
    parser.add_argument("--out-h5", type=Path, default=Path("data/datasets/expert_trajectories.h5"))
    parser.add_argument("--save-h5", action="store_true")
    args = parser.parse_args()

    env = GraspEnv(render_mode="none")
    env.config.max_steps = args.max_steps  # type: ignore[misc]
    expert = ScriptedExpertPolicy(env)

    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    next_states: list[np.ndarray] = []
    rewards: list[float] = []
    dones: list[bool] = []

    successes = 0

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        expert.reset()

        while True:
            action = expert.act(obs)
            nxt, rew, terminated, truncated, info = env.step(action)

            states.append(obs.copy())
            actions.append(action.copy())
            next_states.append(nxt.copy())
            rewards.append(float(rew))
            dones.append(bool(terminated or truncated))

            obs = nxt
            if terminated or truncated:
                successes += int(info.get("success", False))
                break

    env.close()

    states_np = np.asarray(states, dtype=np.float32)
    actions_np = np.asarray(actions, dtype=np.float32)
    next_states_np = np.asarray(next_states, dtype=np.float32)
    rewards_np = np.asarray(rewards, dtype=np.float32)
    dones_np = np.asarray(dones, dtype=bool)

    args.out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_npz,
        states=states_np,
        actions=actions_np,
        next_states=next_states_np,
        rewards=rewards_np,
        dones=dones_np,
    )

    if args.save_h5:
        if h5py is None:
            raise RuntimeError("h5py is not installed. Install h5py or disable --save-h5.")
        args.out_h5.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(args.out_h5, "w") as h:
            h.create_dataset("states", data=states_np)
            h.create_dataset("actions", data=actions_np)
            h.create_dataset("next_states", data=next_states_np)
            h.create_dataset("rewards", data=rewards_np)
            h.create_dataset("dones", data=dones_np)

    success_rate = successes / max(1, args.episodes)
    print(f"Saved expert dataset (npz): {args.out_npz}")
    if args.save_h5:
        print(f"Saved expert dataset (h5): {args.out_h5}")
    print(f"Transitions: {len(states_np)}")
    print(f"Episode success rate: {success_rate:.3f}")


if __name__ == "__main__":
    main()
