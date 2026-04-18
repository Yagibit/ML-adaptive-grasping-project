from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "grip_command",
    "adaptive_final_grip_command",
    "gripper_joint_mean",
    "object_size_x",
    "object_size_y",
    "object_size_z",
    "object_mass",
    "object_initial_z",
    "object_max_z",
    "object_final_z",
    "object_height_delta",
    "contact_count_total_peak",
    "contact_count_obj_gripper_peak",
    "mean_contact_normal_force",
    "tactile_contact_mean",
    "tactile_force_mean",
    "slip_event_count",
]


@dataclass
class ClassificationArtifacts:
    logistic_path: Path
    random_forest_path: Path


@dataclass
class RegressionArtifacts:
    random_forest_path: Path
    torch_model_path: Path
    torch_scaler_path: Path


class MLPRegressor(torch.nn.Module):
    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_features, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _numeric_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                FEATURE_COLUMNS,
            )
        ],
        remainder="drop",
    )


def train_classification(df: pd.DataFrame, artifacts_dir: Path, seed: int = 42) -> tuple[dict[str, float], ClassificationArtifacts]:
    x = df[FEATURE_COLUMNS]
    y = df["success"].astype(int)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=seed,
        stratify=y if y.nunique() > 1 else None,
    )

    logistic = Pipeline(
        steps=[
            ("prep", _numeric_preprocessor()),
            ("model", LogisticRegression(max_iter=2000)),
        ]
    )
    logistic.fit(x_train, y_train)
    logistic_pred = logistic.predict(x_test)

    logistic_metrics: dict[str, float] = {
        "logistic_accuracy": float(accuracy_score(y_test, logistic_pred)),
        "logistic_precision": float(precision_score(y_test, logistic_pred, zero_division=0)),
        "logistic_recall": float(recall_score(y_test, logistic_pred, zero_division=0)),
        "logistic_f1": float(f1_score(y_test, logistic_pred, zero_division=0)),
    }
    if y_test.nunique() > 1:
        logistic_prob = logistic.predict_proba(x_test)[:, 1]
        logistic_metrics["logistic_roc_auc"] = float(roc_auc_score(y_test, logistic_prob))

    rf = Pipeline(
        steps=[
            ("prep", _numeric_preprocessor()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=None,
                    min_samples_leaf=2,
                    random_state=seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    rf.fit(x_train, y_train)
    rf_pred = rf.predict(x_test)

    rf_metrics: dict[str, float] = {
        "rf_accuracy": float(accuracy_score(y_test, rf_pred)),
        "rf_precision": float(precision_score(y_test, rf_pred, zero_division=0)),
        "rf_recall": float(recall_score(y_test, rf_pred, zero_division=0)),
        "rf_f1": float(f1_score(y_test, rf_pred, zero_division=0)),
    }
    if y_test.nunique() > 1:
        rf_prob = rf.predict_proba(x_test)[:, 1]
        rf_metrics["rf_roc_auc"] = float(roc_auc_score(y_test, rf_prob))

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    logistic_path = artifacts_dir / "logistic_success.joblib"
    rf_path = artifacts_dir / "random_forest_success.joblib"
    joblib.dump(logistic, logistic_path)
    joblib.dump(rf, rf_path)

    metrics = {}
    metrics.update(logistic_metrics)
    metrics.update(rf_metrics)

    return metrics, ClassificationArtifacts(logistic_path=logistic_path, random_forest_path=rf_path)


def _torch_regression_train(x_train: np.ndarray, y_train: np.ndarray, seed: int) -> MLPRegressor:
    torch.manual_seed(seed)
    model = MLPRegressor(n_features=x_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()

    x_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)

    for _ in range(350):
        optimizer.zero_grad()
        pred = model(x_tensor)
        loss = loss_fn(pred, y_tensor)
        loss.backward()
        optimizer.step()

    return model


def train_regression(df: pd.DataFrame, artifacts_dir: Path, seed: int = 42) -> tuple[dict[str, float], RegressionArtifacts]:
    reg_df = df.dropna(subset=["optimal_grip_command"]).copy()
    if reg_df.empty:
        raise ValueError("No rows with optimal_grip_command available. Generate data with successful grasps first.")

    x = reg_df[FEATURE_COLUMNS]
    y = reg_df["optimal_grip_command"].astype(float)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)

    rf = Pipeline(
        steps=[
            ("prep", _numeric_preprocessor()),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=400,
                    max_depth=None,
                    min_samples_leaf=2,
                    random_state=seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    rf.fit(x_train, y_train)
    rf_pred = rf.predict(x_test)

    rf_metrics = {
        "rf_reg_mae": float(mean_absolute_error(y_test, rf_pred)),
        "rf_reg_rmse": float(np.sqrt(mean_squared_error(y_test, rf_pred))),
        "rf_reg_r2": float(r2_score(y_test, rf_pred)),
    }

    prep = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    x_train_scaled = prep.fit_transform(x_train)
    x_test_scaled = prep.transform(x_test)

    torch_model = _torch_regression_train(x_train_scaled, y_train.to_numpy(), seed=seed)
    torch_model.eval()
    with torch.no_grad():
        torch_pred = torch_model(torch.tensor(x_test_scaled, dtype=torch.float32)).cpu().numpy().reshape(-1)

    torch_metrics = {
        "torch_reg_mae": float(mean_absolute_error(y_test, torch_pred)),
        "torch_reg_rmse": float(np.sqrt(mean_squared_error(y_test, torch_pred))),
        "torch_reg_r2": float(r2_score(y_test, torch_pred)),
    }

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    rf_path = artifacts_dir / "random_forest_optimal_force.joblib"
    torch_path = artifacts_dir / "torch_mlp_optimal_force.pt"
    scaler_path = artifacts_dir / "torch_mlp_scaler.joblib"
    joblib.dump(rf, rf_path)
    torch.save(torch_model.state_dict(), torch_path)
    joblib.dump(prep, scaler_path)

    metrics = {}
    metrics.update(rf_metrics)
    metrics.update(torch_metrics)

    return metrics, RegressionArtifacts(
        random_forest_path=rf_path,
        torch_model_path=torch_path,
        torch_scaler_path=scaler_path,
    )
