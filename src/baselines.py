from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import apply_budgeted_selection


def all_gpu_policy(feature_df: pd.DataFrame, memory_budget_mb: float) -> pd.DataFrame:
    decisions = feature_df.copy()
    decisions["placement"] = "gpu"
    decisions["policy_name"] = "all_gpu"
    decisions["memory_budget_mb"] = memory_budget_mb
    decisions["placement_score"] = 1.0
    return decisions


def overflow_to_cpu_policy(feature_df: pd.DataFrame, memory_budget_mb: float) -> pd.DataFrame:
    working = feature_df.copy()
    type_priority = {"weight": 3.0, "kv_cache": 2.0, "activation": 1.0}
    working["placement_score"] = (
        working["tensor_type"].map(type_priority).fillna(0.0) * 1_000.0
        + (working["prompt_length"] * 0.05)
        - (working["size_mb"] * 0.10)
        - (working["layer_id"] * 0.01)
    )
    return apply_budgeted_selection(
        working,
        score_column="placement_score",
        memory_budget_mb=memory_budget_mb,
        policy_name="overflow_to_cpu",
    )


def heuristic_policy(feature_df: pd.DataFrame, memory_budget_mb: float) -> pd.DataFrame:
    working = feature_df.copy()
    type_bonus = np.select(
        [
            working["tensor_type"] == "weight",
            working["tensor_type"] == "kv_cache",
            working["tensor_type"] == "activation",
        ],
        [1.15, 1.30, 0.85],
        default=1.0,
    )
    working["placement_score"] = (
        (working["access_count"] + working["touch_count"])
        * (1.0 + 1.50 * working["decode_fraction"])
        * type_bonus
        / working["size_mb"].clip(lower=0.25)
    )
    return apply_budgeted_selection(
        working,
        score_column="placement_score",
        memory_budget_mb=memory_budget_mb,
        policy_name="heuristic",
    )


def available_baselines() -> dict[str, callable]:
    return {
        "all_gpu": all_gpu_policy,
        "overflow_to_cpu": overflow_to_cpu_policy,
        "heuristic": heuristic_policy,
    }
