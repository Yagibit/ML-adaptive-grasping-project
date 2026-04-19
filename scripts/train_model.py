from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


FEATURES = ["wrist_ctrl", "hand_ctrl", "contact_count"]
TARGET = "success"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RandomForestClassifier for grasp success")
    parser.add_argument("--dataset", type=Path, default=Path("data/grip_dataset.csv"))
    parser.add_argument("--output", type=Path, default=Path("models/trained_model.pkl"))
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    dataset_path = args.dataset if args.dataset.is_absolute() else project_root / args.dataset
    output_path = args.output if args.output.is_absolute() else project_root / args.output

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path)
    if df.empty:
        raise ValueError("Dataset is empty.")

    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    x = df[FEATURES]
    y = df[TARGET].astype(int)

    model = RandomForestClassifier(n_estimators=250, random_state=42)
    model.fit(x, y)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    print(f"Saved model: {output_path}")
    print(f"Training rows: {len(df)}")


if __name__ == "__main__":
    main()
