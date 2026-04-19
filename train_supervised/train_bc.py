from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from data.trajectory_dataset import TrajectoryDataset
from models.policy_network import PolicyMLP


def main() -> None:
    parser = argparse.ArgumentParser(description="Train behavior cloning policy from expert data")
    parser.add_argument("--dataset", type=Path, default=Path("data/datasets/expert_trajectories.npz"))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("models/checkpoints/bc_policy.pt"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset = TrajectoryDataset(args.dataset)
    n_total = len(dataset)
    n_val = max(1, int(0.1 * n_total))
    n_train = n_total - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    obs_dim = dataset.states.shape[1]
    action_dim = dataset.actions.shape[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PolicyMLP(obs_dim, action_dim).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for s, a, _ in train_loader:
            s = s.to(device)
            a = a.to(device)
            pred = model(s)
            loss = loss_fn(pred, a)
            optim.zero_grad()
            loss.backward()
            optim.step()
            train_losses.append(float(loss.item()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for s, a, _ in val_loader:
                s = s.to(device)
                a = a.to(device)
                pred = model(s)
                val_losses.append(float(loss_fn(pred, a).item()))

        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        val_loss = float(np.mean(val_losses)) if val_losses else 0.0
        print(f"epoch={epoch:03d} train_mse={train_loss:.6f} val_mse={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            args.out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "obs_dim": obs_dim,
                    "action_dim": action_dim,
                    "val_mse": best_val,
                },
                args.out,
            )

    print(f"Saved best BC policy: {args.out}")


if __name__ == "__main__":
    main()
