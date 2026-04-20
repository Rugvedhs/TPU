from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _prepare_figure(style: str = "ggplot", dpi: int = 160) -> None:
    plt.style.use(style)
    plt.rcParams["figure.dpi"] = dpi


def plot_kv_cache_growth(events_df: pd.DataFrame, output_path: str | Path, *, style: str = "ggplot", dpi: int = 160) -> Path:
    _prepare_figure(style=style, dpi=dpi)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    kv = events_df[events_df["tensor_type"] == "kv_cache"].copy()
    growth = (
        kv.groupby("step_id", as_index=False)
        .agg(kv_cache_mb=("size_bytes", lambda values: values.sum() / (1024 ** 2)))
        .sort_values("step_id")
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(growth["step_id"], growth["kv_cache_mb"], linewidth=2.0, color="#0f766e")
    ax.set_title("KV-Cache Growth Over Time")
    ax.set_xlabel("Decode Step")
    ax.set_ylabel("Aggregate KV Cache (MB)")
    fig.tight_layout()
    fig.savefig(target)
    plt.close(fig)
    return target


def plot_access_frequency_by_layer(events_df: pd.DataFrame, output_path: str | Path, *, style: str = "ggplot", dpi: int = 160) -> Path:
    _prepare_figure(style=style, dpi=dpi)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    summary = (
        events_df.groupby(["layer_id", "tensor_type"], as_index=False)
        .agg(access_count=("access_count", "sum"))
        .sort_values(["layer_id", "tensor_type"])
    )
    pivot = summary.pivot(index="layer_id", columns="tensor_type", values="access_count").fillna(0.0)
    fig, ax = plt.subplots(figsize=(8, 4))
    pivot.plot(kind="bar", stacked=True, ax=ax, color=["#1d4ed8", "#16a34a", "#b45309"])
    ax.set_title("Access Frequency by Layer")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Access Count")
    fig.tight_layout()
    fig.savefig(target)
    plt.close(fig)
    return target


def plot_metric_vs_budget(
    results_df: pd.DataFrame,
    *,
    metric: str,
    ylabel: str,
    output_path: str | Path,
    style: str = "ggplot",
    dpi: int = 160,
) -> Path:
    _prepare_figure(style=style, dpi=dpi)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    for method, method_frame in results_df.groupby("policy_name", sort=False):
        ax.plot(
            method_frame["memory_budget_mb"],
            method_frame[metric],
            marker="o",
            linewidth=2.0,
            label=method,
        )
    ax.set_title(f"{ylabel} vs Memory Budget")
    ax.set_xlabel("Memory Budget (MB)")
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    fig.savefig(target)
    plt.close(fig)
    return target


def plot_method_comparison(results_df: pd.DataFrame, output_path: str | Path, *, style: str = "ggplot", dpi: int = 160) -> Path:
    _prepare_figure(style=style, dpi=dpi)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    pivot = results_df.pivot(index="memory_budget_mb", columns="policy_name", values="throughput_tps")
    fig, ax = plt.subplots(figsize=(7, 4))
    pivot.plot(ax=ax, marker="o", linewidth=2.0)
    ax.set_title("Learned Policy vs Baselines")
    ax.set_xlabel("Memory Budget (MB)")
    ax.set_ylabel("Throughput (tokens/sec)")
    fig.tight_layout()
    fig.savefig(target)
    plt.close(fig)
    return target
