from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import ProjectConfig
from src.features import FEATURE_COLUMNS, build_training_frame
from src.policy import PlacementMLP, fit_standardizer, predict_policy_placements, save_policy, transform_features
from src.simulator import simulate_trace


def _split_runs(feature_df: pd.DataFrame, validation_fraction: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique_runs = feature_df["run_id"].dropna().unique().tolist()
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_runs)
    if len(unique_runs) <= 1:
        cutoff = max(1, int(len(feature_df) * (1.0 - validation_fraction)))
        return feature_df.iloc[:cutoff].copy(), feature_df.iloc[cutoff:].copy()

    val_count = max(1, int(len(unique_runs) * validation_fraction))
    val_runs = set(unique_runs[:val_count])
    train_df = feature_df[~feature_df["run_id"].isin(val_runs)].copy()
    val_df = feature_df[feature_df["run_id"].isin(val_runs)].copy()
    if val_df.empty:
        val_df = train_df.sample(frac=validation_fraction, random_state=seed)
        train_df = train_df.drop(index=val_df.index)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def train_policy(
    config: ProjectConfig,
    events_df: pd.DataFrame,
    *,
    output_dir: str | Path | None = None,
    memory_budget_mb: float | None = None,
) -> dict[str, Path]:
    budget_mb = float(memory_budget_mb or config.training.train_budget_mb)
    feature_df = build_training_frame(events_df, memory_budget_mb=budget_mb)
    if feature_df.empty:
        raise ValueError("No training data could be built from the provided trace.")

    train_df, val_df = _split_runs(feature_df, config.training.validation_fraction, config.seed)
    if val_df.empty:
        val_df = train_df.copy()

    scaler = fit_standardizer(train_df, FEATURE_COLUMNS)
    train_x = torch.tensor(transform_features(train_df, scaler, FEATURE_COLUMNS), dtype=torch.float32)
    train_y = torch.tensor(train_df["target_gpu"].to_numpy(dtype=np.float32), dtype=torch.float32)
    val_x = torch.tensor(transform_features(val_df, scaler, FEATURE_COLUMNS), dtype=torch.float32)
    val_y = torch.tensor(val_df["target_gpu"].to_numpy(dtype=np.float32), dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=config.training.batch_size,
        shuffle=True,
    )

    model = PlacementMLP(
        input_dim=len(FEATURE_COLUMNS),
        hidden_dim=config.training.hidden_dim,
        dropout=config.training.dropout,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss()

    history: list[dict[str, float | int]] = []
    best_val_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None

    val_run_ids = val_df["run_id"].dropna().unique().tolist()
    val_events = events_df[events_df["run_id"].isin(val_run_ids)].copy()

    for epoch in range(1, config.training.epochs + 1):
        model.train()
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * len(batch_x)

        model.eval()
        with torch.no_grad():
            val_logits = model(val_x)
            val_loss = float(criterion(val_logits, val_y).item())
            val_probs = torch.sigmoid(val_logits)
            val_pred = (val_probs >= 0.5).float()
            val_acc = float((val_pred == val_y).float().mean().item())

        val_decisions = predict_policy_placements(
            model,
            val_df,
            scaler,
            memory_budget_mb=budget_mb,
            feature_columns=FEATURE_COLUMNS,
        )
        val_reward = 0.0
        if not val_events.empty and not val_decisions.empty:
            metrics = simulate_trace(
                val_events,
                val_decisions,
                memory_budget_mb=budget_mb,
                transfer_cost_per_mb_ms=config.benchmark.transfer_cost_per_mb_ms,
                decode_transfer_multiplier=config.benchmark.decode_transfer_multiplier,
                kv_cache_transfer_multiplier=config.benchmark.kv_cache_transfer_multiplier,
                cpu_resident_penalty_ms=config.benchmark.cpu_resident_penalty_ms,
                oom_penalty_ms_per_mb=config.benchmark.oom_penalty_ms_per_mb,
            )
            val_reward = float(metrics["throughput_tps"])

        epoch_record = {
            "epoch": epoch,
            "train_loss": running_loss / max(len(train_df), 1),
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_reward": val_reward,
        }
        history.append(epoch_record)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    target_dir = Path(output_dir) if output_dir else config.paths.results_logs
    target_dir.mkdir(parents=True, exist_ok=True)

    history_path = target_dir / "policy_training_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)

    metadata = {
        "memory_budget_mb": budget_mb,
        "seed": config.seed,
        "epochs": config.training.epochs,
    }
    model_path = save_policy(
        model,
        scaler,
        target_dir / "learned_policy.pt",
        feature_columns=FEATURE_COLUMNS,
        metadata=metadata,
    )

    decision_examples = predict_policy_placements(
        model,
        feature_df,
        scaler,
        memory_budget_mb=budget_mb,
        feature_columns=FEATURE_COLUMNS,
    )
    decision_examples = decision_examples.merge(
        feature_df[["run_id", "tensor_name", "oracle_placement", "target_gpu", "tensor_type", "layer_id", "size_mb", "access_count"]],
        on=["run_id", "tensor_name", "tensor_type", "layer_id", "size_mb", "access_count"],
        how="left",
    )
    decisions_path = target_dir / "decision_examples.csv"
    decision_examples.to_csv(decisions_path, index=False)

    metadata_path = target_dir / "training_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    feature_frame_path = target_dir / "training_features.csv"
    feature_df.to_csv(feature_frame_path, index=False)

    return {
        "model_path": model_path,
        "history_path": history_path,
        "decisions_path": decisions_path,
        "metadata_path": metadata_path,
        "feature_frame_path": feature_frame_path,
    }
