from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor

from env.grasp_env import GraspEnv
from models.policy_network import PolicyMLP


def _init_ppo_from_bc(model: PPO, bc_path: Path) -> None:
    if not bc_path.exists():
        return

    blob = torch.load(bc_path, map_location="cpu")
    bc = PolicyMLP(blob["obs_dim"], blob["action_dim"])
    bc.load_state_dict(blob["model_state"])

    try:
        ppo_policy = model.policy
        ppo_policy.mlp_extractor.policy_net[0].weight.data.copy_(bc.backbone[0].weight.data)
        ppo_policy.mlp_extractor.policy_net[0].bias.data.copy_(bc.backbone[0].bias.data)
        ppo_policy.mlp_extractor.policy_net[2].weight.data.copy_(bc.backbone[2].weight.data)
        ppo_policy.mlp_extractor.policy_net[2].bias.data.copy_(bc.backbone[2].bias.data)
        ppo_policy.action_net.weight.data.copy_(bc.mean_head.weight.data)
        ppo_policy.action_net.bias.data.copy_(bc.mean_head.bias.data)
        print("Initialized PPO policy from BC weights.")
    except Exception as exc:  # best-effort init
        print(f"Warning: could not map BC weights to PPO policy: {exc}")


def _init_sac_from_bc(model: SAC, bc_path: Path) -> None:
    if not bc_path.exists():
        return

    blob = torch.load(bc_path, map_location="cpu")
    bc = PolicyMLP(blob["obs_dim"], blob["action_dim"])
    bc.load_state_dict(blob["model_state"])

    try:
        actor = model.policy.actor
        actor.latent_pi[0].weight.data.copy_(bc.backbone[0].weight.data)
        actor.latent_pi[0].bias.data.copy_(bc.backbone[0].bias.data)
        actor.latent_pi[2].weight.data.copy_(bc.backbone[2].weight.data)
        actor.latent_pi[2].bias.data.copy_(bc.backbone[2].bias.data)
        actor.mu.weight.data.copy_(bc.mean_head.weight.data)
        actor.mu.bias.data.copy_(bc.mean_head.bias.data)
        print("Initialized SAC actor from BC weights.")
    except Exception as exc:  # best-effort init
        print(f"Warning: could not map BC weights to SAC actor: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RL refinement (PPO/SAC) starting from BC policy")
    parser.add_argument("--algo", choices=["ppo", "sac"], default="ppo")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bc-path", type=Path, default=Path("models/checkpoints/bc_policy.pt"))
    parser.add_argument("--out", type=Path, default=Path("models/rl/rl_policy.zip"))
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = Monitor(GraspEnv(render_mode="none"))

    if args.algo == "ppo":
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs={"net_arch": {"pi": [256, 256], "vf": [256, 256]}, "activation_fn": torch.nn.ReLU},
            n_steps=1024,
            batch_size=256,
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=1,
            seed=args.seed,
        )
        _init_ppo_from_bc(model, args.bc_path)
    else:
        model = SAC(
            "MlpPolicy",
            env,
            policy_kwargs={"net_arch": [256, 256]},
            learning_rate=3e-4,
            batch_size=256,
            gamma=0.99,
            verbose=1,
            seed=args.seed,
        )
        _init_sac_from_bc(model, args.bc_path)

    model.learn(total_timesteps=args.timesteps, progress_bar=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(args.out.with_suffix("")))
    print(f"Saved RL policy: {args.out}")


if __name__ == "__main__":
    main()
