from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.features import FEATURE_COLUMNS, apply_budgeted_selection, feature_matrix


class PlacementMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.10) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs).squeeze(-1)


def fit_standardizer(frame: pd.DataFrame, feature_columns: list[str] | None = None) -> dict[str, np.ndarray]:
    columns = feature_columns or FEATURE_COLUMNS
    values = feature_matrix(frame, columns)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std[std < 1e-6] = 1.0
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}


def transform_features(
    frame: pd.DataFrame,
    scaler: dict[str, np.ndarray],
    feature_columns: list[str] | None = None,
) -> np.ndarray:
    columns = feature_columns or FEATURE_COLUMNS
    values = feature_matrix(frame, columns)
    return (values - scaler["mean"]) / scaler["std"]


@torch.no_grad()
def predict_scores(
    model: PlacementMLP,
    frame: pd.DataFrame,
    scaler: dict[str, np.ndarray],
    feature_columns: list[str] | None = None,
    device: str = "cpu",
) -> np.ndarray:
    columns = feature_columns or FEATURE_COLUMNS
    model = model.to(device)
    model.eval()
    inputs = torch.tensor(transform_features(frame, scaler, columns), dtype=torch.float32, device=device)
    logits = model(inputs)
    return torch.sigmoid(logits).cpu().numpy()


def predict_policy_placements(
    model: PlacementMLP,
    frame: pd.DataFrame,
    scaler: dict[str, np.ndarray],
    memory_budget_mb: float,
    feature_columns: list[str] | None = None,
    device: str = "cpu",
) -> pd.DataFrame:
    scores = predict_scores(model, frame, scaler, feature_columns=feature_columns, device=device)
    working = frame.copy()
    working["placement_score"] = scores
    return apply_budgeted_selection(
        working,
        score_column="placement_score",
        memory_budget_mb=memory_budget_mb,
        policy_name="learned_policy",
    )


def save_policy(
    model: PlacementMLP,
    scaler: dict[str, np.ndarray],
    output_path: str | Path,
    feature_columns: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "scaler_mean": scaler["mean"],
        "scaler_std": scaler["std"],
        "feature_columns": feature_columns or FEATURE_COLUMNS,
        "hidden_dim": getattr(model, "hidden_dim", 64),
        "metadata": metadata or {},
    }
    torch.save(payload, target)
    return target


def load_policy(path: str | Path, device: str = "cpu") -> tuple[PlacementMLP, dict[str, np.ndarray], list[str], dict[str, Any]]:
    load_kwargs = {"map_location": device}
    try:
        # PyTorch 2.6+ defaults to weights_only=True, which rejects our metadata-rich checkpoints.
        payload = torch.load(Path(path), weights_only=False, **load_kwargs)
    except TypeError:
        payload = torch.load(Path(path), **load_kwargs)
    feature_columns = list(payload["feature_columns"])
    hidden_dim = int(payload.get("hidden_dim", 64))
    model = PlacementMLP(input_dim=len(feature_columns), hidden_dim=hidden_dim)
    model.load_state_dict(payload["state_dict"])
    scaler = {
        "mean": np.asarray(payload["scaler_mean"], dtype=np.float32),
        "std": np.asarray(payload["scaler_std"], dtype=np.float32),
    }
    metadata = dict(payload.get("metadata", {}))
    return model, scaler, feature_columns, metadata
