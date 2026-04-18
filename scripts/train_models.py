from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from _bootstrap import add_project_root_to_path

add_project_root_to_path()

from models.training_pipeline import FEATURE_COLUMNS, train_classification, train_regression


def _validate_columns(df: pd.DataFrame) -> None:
    required = set(FEATURE_COLUMNS + ["success", "optimal_grip_command"])
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train supervised models for adaptive grip-force control")
    parser.add_argument("--dataset", type=Path, default=Path("data/raw/grip_dataset.csv"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("models/artifacts"))
    parser.add_argument("--metrics-out", type=Path, default=Path("models/artifacts/metrics.json"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset not found at {args.dataset}")

    df = pd.read_csv(args.dataset)
    if df.empty:
        raise ValueError("Dataset is empty. Generate trials before training.")

    _validate_columns(df)

    cls_metrics, cls_artifacts = train_classification(df, args.artifacts_dir, seed=args.seed)
    reg_metrics, reg_artifacts = train_regression(df, args.artifacts_dir, seed=args.seed)

    metrics = {
        "classification": cls_metrics,
        "regression": reg_metrics,
        "artifacts": {
            "logistic": str(cls_artifacts.logistic_path),
            "rf_classifier": str(cls_artifacts.random_forest_path),
            "rf_regressor": str(reg_artifacts.random_forest_path),
            "torch_regressor": str(reg_artifacts.torch_model_path),
            "torch_scaler": str(reg_artifacts.torch_scaler_path),
        },
    }

    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Training completed.")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
