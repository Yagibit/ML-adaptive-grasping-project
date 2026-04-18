from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for plotting: {missing}")


def _plot_contact_force_over_trials(df: pd.DataFrame, out_path: Path) -> None:
    _require_columns(df, ["trial_id", "tactile_contact_mean", "tactile_force_mean"])

    trial_df = (
        df[["trial_id", "tactile_contact_mean", "tactile_force_mean"]]
        .sort_values("trial_id")
        .reset_index(drop=True)
    )

    plt.figure(figsize=(11, 5))
    plt.plot(trial_df["trial_id"], trial_df["tactile_contact_mean"], label="Tactile Contact Mean", linewidth=1.5)
    plt.plot(trial_df["trial_id"], trial_df["tactile_force_mean"], label="Tactile Force Mean", linewidth=1.5)
    plt.xlabel("Trial ID (time order)")
    plt.ylabel("Feedback Value")
    plt.title("Tactile Contact and Force Over Trial Time")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_success_vs_adaptive_force(df: pd.DataFrame, out_path: Path) -> None:
    _require_columns(df, ["adaptive_final_grip_command", "success"])

    plt.figure(figsize=(8, 5))
    success0 = df[df["success"] == 0]["adaptive_final_grip_command"]
    success1 = df[df["success"] == 1]["adaptive_final_grip_command"]

    bins = 24
    plt.hist(success0, bins=bins, alpha=0.6, label="Failure (0)", density=True)
    plt.hist(success1, bins=bins, alpha=0.6, label="Success (1)", density=True)
    plt.xlabel("Adaptive Final Grip Command")
    plt.ylabel("Density")
    plt.title("Success vs Adaptive Final Grip Command")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_slip_distribution(df: pd.DataFrame, out_path: Path) -> None:
    _require_columns(df, ["slip_event_count", "success"])

    counts_fail = df[df["success"] == 0]["slip_event_count"]
    counts_succ = df[df["success"] == 1]["slip_event_count"]

    upper = int(max(df["slip_event_count"].max(), 1))
    bins = list(range(0, upper + 2))

    plt.figure(figsize=(8, 5))
    plt.hist(counts_fail, bins=bins, alpha=0.6, label="Failure (0)")
    plt.hist(counts_succ, bins=bins, alpha=0.6, label="Success (1)")
    plt.xlabel("Slip Event Count")
    plt.ylabel("Frequency")
    plt.title("Slip Event Distribution by Outcome")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_tactile_feedback_plots(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    p1 = out_dir / "contact_force_over_trials.png"
    p2 = out_dir / "success_vs_adaptive_final_grip_command.png"
    p3 = out_dir / "slip_event_distribution.png"

    _plot_contact_force_over_trials(df, p1)
    _plot_success_vs_adaptive_force(df, p2)
    _plot_slip_distribution(df, p3)

    return [p1, p2, p3]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate tactile-feedback plots from grip dataset")
    parser.add_argument("--dataset", type=Path, default=Path("data/raw/grip_dataset.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/plots"))
    args = parser.parse_args()

    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")

    df = pd.read_csv(args.dataset)
    if df.empty:
        raise ValueError("Dataset is empty, cannot generate plots.")

    paths = save_tactile_feedback_plots(df, args.out_dir)
    print("Saved tactile-feedback plots:")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
