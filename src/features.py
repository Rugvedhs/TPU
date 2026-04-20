from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "size_mb",
    "layer_id",
    "prompt_length",
    "batch_size",
    "access_count",
    "touch_count",
    "mean_latency_ms",
    "max_latency_ms",
    "decode_fraction",
    "prefill_fraction",
    "is_weight",
    "is_kv_cache",
    "is_activation",
    "hotness",
    "pressure_score",
]


def aggregate_tensor_events(events_df: pd.DataFrame) -> pd.DataFrame:
    if events_df.empty:
        return pd.DataFrame()

    working = events_df.copy()
    if "size_mb" not in working.columns:
        working["size_mb"] = working["size_bytes"] / (1024 ** 2)

    grouped = (
        working.groupby(["run_id", "tensor_name", "tensor_type", "layer_id"], as_index=False)
        .agg(
            size_mb=("size_mb", "max"),
            access_count=("access_count", "sum"),
            touch_count=("tensor_name", "size"),
            mean_latency_ms=("latency_ms", "mean"),
            max_latency_ms=("latency_ms", "max"),
            prompt_length=("prompt_length", "max"),
            batch_size=("batch_size", "max"),
            decode_touches=("phase", lambda values: int((values == "decode").sum())),
            prefill_touches=("phase", lambda values: int((values == "prefill").sum())),
        )
        .sort_values(["run_id", "layer_id", "tensor_name"])
        .reset_index(drop=True)
    )

    total_touches = (grouped["decode_touches"] + grouped["prefill_touches"]).clip(lower=1)
    grouped["decode_fraction"] = grouped["decode_touches"] / total_touches
    grouped["prefill_fraction"] = grouped["prefill_touches"] / total_touches
    grouped["is_weight"] = (grouped["tensor_type"] == "weight").astype(float)
    grouped["is_kv_cache"] = (grouped["tensor_type"] == "kv_cache").astype(float)
    grouped["is_activation"] = (grouped["tensor_type"] == "activation").astype(float)
    grouped["hotness"] = grouped["access_count"] / grouped["size_mb"].clip(lower=0.25)
    grouped["pressure_score"] = grouped["size_mb"] * (1.0 + grouped["decode_fraction"])
    return grouped


def score_for_oracle(feature_df: pd.DataFrame) -> pd.Series:
    type_bonus = np.select(
        [
            feature_df["tensor_type"] == "weight",
            feature_df["tensor_type"] == "kv_cache",
            feature_df["tensor_type"] == "activation",
        ],
        [1.20, 1.35, 0.90],
        default=1.0,
    )
    score = (
        (feature_df["access_count"] + feature_df["touch_count"])
        * (1.0 + feature_df["decode_fraction"])
        * (1.0 + feature_df["mean_latency_ms"])
        * type_bonus
        / feature_df["size_mb"].clip(lower=0.25)
    )
    return score


def apply_budgeted_selection(
    frame: pd.DataFrame,
    *,
    score_column: str,
    memory_budget_mb: float,
    policy_name: str,
) -> pd.DataFrame:
    selections: list[pd.DataFrame] = []
    for run_id, run_frame in frame.groupby("run_id", sort=False):
        ordered = run_frame.sort_values(score_column, ascending=False).copy()
        ordered["cumulative_mb"] = ordered["size_mb"].cumsum()
        ordered["placement"] = np.where(ordered["cumulative_mb"] <= memory_budget_mb, "gpu", "cpu")
        ordered["policy_name"] = policy_name
        ordered["memory_budget_mb"] = memory_budget_mb
        ordered["run_id"] = run_id
        selections.append(ordered)
    return pd.concat(selections, ignore_index=True) if selections else pd.DataFrame()


def build_training_frame(events_df: pd.DataFrame, memory_budget_mb: float) -> pd.DataFrame:
    feature_df = aggregate_tensor_events(events_df)
    if feature_df.empty:
        return feature_df
    feature_df = feature_df.copy()
    feature_df["oracle_score"] = score_for_oracle(feature_df)
    oracle = apply_budgeted_selection(
        feature_df,
        score_column="oracle_score",
        memory_budget_mb=memory_budget_mb,
        policy_name="oracle",
    )
    merged = feature_df.merge(
        oracle[["run_id", "tensor_name", "placement"]],
        on=["run_id", "tensor_name"],
        how="left",
        suffixes=("", "_oracle"),
    )
    merged["target_gpu"] = (merged["placement"] == "gpu").astype(float)
    merged = merged.rename(columns={"placement": "oracle_placement"})
    return merged


def feature_matrix(frame: pd.DataFrame, feature_columns: Iterable[str] | None = None) -> np.ndarray:
    columns = list(feature_columns or FEATURE_COLUMNS)
    return frame[columns].fillna(0.0).to_numpy(dtype=np.float32)


def summarize_hot_tensors(feature_df: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    return (
        feature_df.sort_values("hotness", ascending=False)
        .head(top_k)[
            [
                "run_id",
                "tensor_name",
                "tensor_type",
                "layer_id",
                "size_mb",
                "access_count",
                "touch_count",
                "hotness",
            ]
        ]
        .reset_index(drop=True)
    )


def summarize_memory_by_layer(feature_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        feature_df.groupby(["layer_id", "tensor_type"], as_index=False)
        .agg(total_size_mb=("size_mb", "sum"), total_access_count=("access_count", "sum"))
        .sort_values(["layer_id", "tensor_type"])
        .reset_index(drop=True)
    )
    return summary
