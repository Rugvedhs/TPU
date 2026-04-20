from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.baselines import available_baselines
from src.config import ProjectConfig
from src.features import aggregate_tensor_events, summarize_hot_tensors, summarize_memory_by_layer
from src.policy import load_policy, predict_policy_placements
from src.simulator import simulate_trace


def benchmark_policies(
    config: ProjectConfig,
    events_df: pd.DataFrame,
    *,
    policy_path: str | Path | None = None,
    budgets_mb: list[float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_df = aggregate_tensor_events(events_df)
    if feature_df.empty:
        raise ValueError("No tensor events were provided for benchmarking.")

    budgets = budgets_mb or list(config.benchmark.memory_budgets_mb)
    results: list[dict[str, float | int | str | bool]] = []
    placements: list[pd.DataFrame] = []

    learned_bundle = None
    if policy_path is not None:
        learned_bundle = load_policy(policy_path)

    for budget_mb in budgets:
        for name, policy_fn in available_baselines().items():
            decision_df = policy_fn(feature_df, budget_mb)
            placements.append(decision_df)
            metrics = simulate_trace(
                events_df,
                decision_df,
                memory_budget_mb=budget_mb,
                transfer_cost_per_mb_ms=config.benchmark.transfer_cost_per_mb_ms,
                decode_transfer_multiplier=config.benchmark.decode_transfer_multiplier,
                kv_cache_transfer_multiplier=config.benchmark.kv_cache_transfer_multiplier,
                cpu_resident_penalty_ms=config.benchmark.cpu_resident_penalty_ms,
                oom_penalty_ms_per_mb=config.benchmark.oom_penalty_ms_per_mb,
            )
            metrics["policy_name"] = name
            results.append(metrics)

        if learned_bundle is not None:
            model, scaler, feature_columns, _ = learned_bundle
            learned_decisions = predict_policy_placements(
                model,
                feature_df,
                scaler,
                memory_budget_mb=budget_mb,
                feature_columns=feature_columns,
            )
            placements.append(learned_decisions)
            metrics = simulate_trace(
                events_df,
                learned_decisions,
                memory_budget_mb=budget_mb,
                transfer_cost_per_mb_ms=config.benchmark.transfer_cost_per_mb_ms,
                decode_transfer_multiplier=config.benchmark.decode_transfer_multiplier,
                kv_cache_transfer_multiplier=config.benchmark.kv_cache_transfer_multiplier,
                cpu_resident_penalty_ms=config.benchmark.cpu_resident_penalty_ms,
                oom_penalty_ms_per_mb=config.benchmark.oom_penalty_ms_per_mb,
            )
            metrics["policy_name"] = "learned_policy"
            results.append(metrics)

    results_df = pd.DataFrame(results).sort_values(["memory_budget_mb", "policy_name"]).reset_index(drop=True)
    placement_df = pd.concat(placements, ignore_index=True).sort_values(
        ["memory_budget_mb", "policy_name", "run_id", "layer_id", "tensor_name"]
    )
    return results_df, placement_df


def export_summary_tables(
    events_df: pd.DataFrame,
    placements_df: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, Path]:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    feature_df = aggregate_tensor_events(events_df)

    hottest_path = target_dir / "top_hottest_tensors.csv"
    summarize_hot_tensors(feature_df).to_csv(hottest_path, index=False)

    layer_memory_path = target_dir / "memory_by_layer.csv"
    summarize_memory_by_layer(feature_df).to_csv(layer_memory_path, index=False)

    placement_counts = (
        placements_df.groupby(["memory_budget_mb", "policy_name", "placement"], as_index=False)
        .agg(count=("tensor_name", "size"))
        .sort_values(["memory_budget_mb", "policy_name", "placement"])
    )
    placement_counts_path = target_dir / "placement_counts.csv"
    placement_counts.to_csv(placement_counts_path, index=False)

    return {
        "top_hottest_tensors": hottest_path,
        "memory_by_layer": layer_memory_path,
        "placement_counts": placement_counts_path,
    }
