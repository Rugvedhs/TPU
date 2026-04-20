from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.benchmark import benchmark_policies, export_summary_tables
from src.config import ProjectConfig
from src.plot_utils import (
    plot_access_frequency_by_layer,
    plot_kv_cache_growth,
    plot_method_comparison,
    plot_metric_vs_budget,
)


def evaluate_project(
    config: ProjectConfig,
    events_df: pd.DataFrame,
    *,
    policy_path: str | Path | None = None,
    output_table_dir: str | Path | None = None,
    output_figure_dir: str | Path | None = None,
) -> dict[str, Path]:
    table_dir = Path(output_table_dir) if output_table_dir else config.paths.results_tables
    figure_dir = Path(output_figure_dir) if output_figure_dir else config.paths.results_figures
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    results_df, placements_df = benchmark_policies(config, events_df, policy_path=policy_path)
    benchmark_path = table_dir / "benchmark_results.csv"
    results_df.to_csv(benchmark_path, index=False)

    placements_path = table_dir / "placement_decisions.csv"
    placements_df.to_csv(placements_path, index=False)

    summary_paths = export_summary_tables(events_df, placements_df, table_dir)

    kv_plot = plot_kv_cache_growth(
        events_df,
        figure_dir / "kv_cache_growth.png",
        style=config.plot.style,
        dpi=config.plot.dpi,
    )
    access_plot = plot_access_frequency_by_layer(
        events_df,
        figure_dir / "access_frequency_by_layer.png",
        style=config.plot.style,
        dpi=config.plot.dpi,
    )
    throughput_plot = plot_metric_vs_budget(
        results_df,
        metric="throughput_tps",
        ylabel="Throughput (tokens/sec)",
        output_path=figure_dir / "throughput_vs_memory_budget.png",
        style=config.plot.style,
        dpi=config.plot.dpi,
    )
    latency_plot = plot_metric_vs_budget(
        results_df,
        metric="mean_latency_ms",
        ylabel="Mean Event Latency (ms)",
        output_path=figure_dir / "latency_vs_memory_budget.png",
        style=config.plot.style,
        dpi=config.plot.dpi,
    )
    comparison_plot = plot_method_comparison(
        results_df,
        figure_dir / "learned_vs_baselines.png",
        style=config.plot.style,
        dpi=config.plot.dpi,
    )

    outputs = {
        "benchmark_results": benchmark_path,
        "placement_decisions": placements_path,
        "kv_cache_growth_plot": kv_plot,
        "access_frequency_plot": access_plot,
        "throughput_plot": throughput_plot,
        "latency_plot": latency_plot,
        "comparison_plot": comparison_plot,
    }
    outputs.update(summary_paths)
    return outputs
