from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


FEATURES = ["wrist_ctrl", "hand_ctrl", "contact_count"]
TARGET = "success"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained model and generate requested plots")
    parser.add_argument("--dataset", type=Path, default=Path("data/grip_dataset.csv"))
    parser.add_argument("--model", type=Path, default=Path("models/trained_model.pkl"))
    parser.add_argument("--out-dir", type=Path, default=Path("models"))
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    dataset_path = args.dataset if args.dataset.is_absolute() else project_root / args.dataset
    model_path = args.model if args.model.is_absolute() else project_root / args.model
    out_dir = args.out_dir if args.out_dir.is_absolute() else project_root / args.out_dir

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    df = pd.read_csv(dataset_path)
    x = df[FEATURES]
    y = df[TARGET].astype(int)

    test_size = 0.25 if len(df) >= 8 else 0.5
    stratify = y if y.nunique() > 1 else None
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42, stratify=stratify)

    model = joblib.load(model_path)
    y_pred = model.predict(x_test)

    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"F1-score : {f1_score(y_test, y_pred, zero_division=0):.4f}")

    out_dir.mkdir(parents=True, exist_ok=True)

    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax_cm)
    ax_cm.set_title("Confusion Matrix")
    fig_cm.tight_layout()
    cm_path = out_dir / "confusion_matrix.png"
    fig_cm.savefig(cm_path, dpi=160)
    plt.close(fig_cm)

    fig_w, ax_w = plt.subplots(figsize=(6, 4))
    ax_w.scatter(df["wrist_ctrl"], df["success"], alpha=0.6)
    ax_w.set_xlabel("wrist_ctrl")
    ax_w.set_ylabel("success")
    ax_w.set_title("wrist_ctrl vs success")
    fig_w.tight_layout()
    wrist_path = out_dir / "wrist_ctrl_vs_success.png"
    fig_w.savefig(wrist_path, dpi=160)
    plt.close(fig_w)

    fig_c, ax_c = plt.subplots(figsize=(6, 4))
    ax_c.scatter(df["contact_count"], df["success"], alpha=0.6)
    ax_c.set_xlabel("contact_count")
    ax_c.set_ylabel("success")
    ax_c.set_title("contact_count vs success")
    fig_c.tight_layout()
    contact_path = out_dir / "contact_count_vs_success.png"
    fig_c.savefig(contact_path, dpi=160)
    plt.close(fig_c)

    print(f"Saved: {cm_path}")
    print(f"Saved: {wrist_path}")
    print(f"Saved: {contact_path}")


if __name__ == "__main__":
    main()
