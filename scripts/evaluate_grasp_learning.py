from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO, SAC

from env.expert_policy import ScriptedExpertPolicy
from env.grasp_env import GraspEnv
from models.policy_network import PolicyMLP


def eval_bc(path: Path, episodes: int, max_steps: int, seed: int) -> float:
    env = GraspEnv(render_mode="none")
    env.config.max_steps = max_steps  # type: ignore[misc]

    blob = torch.load(path, map_location="cpu")
    model = PolicyMLP(blob["obs_dim"], blob["action_dim"])
    model.load_state_dict(blob["model_state"])
    model.eval()

    success = 0
    with torch.no_grad():
        for ep in range(episodes):
            obs, _ = env.reset(seed=seed + ep)
            while True:
                inp = torch.from_numpy(obs).float().unsqueeze(0)
                act = model(inp).squeeze(0).numpy()
                act = np.clip(act, env.action_low, env.action_high)
                obs, _, terminated, truncated, info = env.step(act)
                if terminated or truncated:
                    success += int(info.get("success", False))
                    break

    env.close()
    return success / max(1, episodes)


def eval_rl(path: Path, algo: str, episodes: int, max_steps: int, seed: int) -> float:
    env = GraspEnv(render_mode="none")
    env.config.max_steps = max_steps  # type: ignore[misc]

    load_path = str(path.with_suffix(""))
    if algo == "ppo":
        model = PPO.load(load_path)
    else:
        model = SAC.load(load_path)

    success = 0
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                success += int(info.get("success", False))
                break

    env.close()
    return success / max(1, episodes)


def eval_expert(episodes: int, max_steps: int, seed: int) -> float:
    env = GraspEnv(render_mode="none")
    env.config.max_steps = max_steps  # type: ignore[misc]
    expert = ScriptedExpertPolicy(env)

    success = 0
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        expert.reset()
        while True:
            action = expert.act(obs)
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                success += int(info.get("success", False))
                break

    env.close()
    return success / max(1, episodes)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate grasp success rate for expert, BC, or RL policy")
    parser.add_argument("--mode", choices=["expert", "bc", "rl"], default="expert")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bc-path", type=Path, default=Path("models/checkpoints/bc_policy.pt"))
    parser.add_argument("--rl-path", type=Path, default=Path("models/rl/rl_policy.zip"))
    parser.add_argument("--algo", choices=["ppo", "sac"], default="ppo")
    args = parser.parse_args()

    if args.mode == "expert":
        rate = eval_expert(args.episodes, args.max_steps, args.seed)
    elif args.mode == "bc":
        rate = eval_bc(args.bc_path, args.episodes, args.max_steps, args.seed)
    else:
        rate = eval_rl(args.rl_path, args.algo, args.episodes, args.max_steps, args.seed)

    print(f"mode={args.mode} success_rate={rate:.3f}")


if __name__ == "__main__":
    main()
