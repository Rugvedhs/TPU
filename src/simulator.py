from __future__ import annotations

from datetime import datetime
from pathlib import Path
import math

import numpy as np
import pandas as pd

from src.config import ProjectConfig
from src.logger import TensorLogger


MB = 1024 ** 2


def generate_synthetic_traces(
    config: ProjectConfig,
    *,
    num_runs: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(config.seed)
    logger = TensorLogger()
    run_total = int(num_runs or config.profiling.synthetic_runs)

    for run_index in range(run_total):
        prompt_length = int(rng.choice(config.profiling.prompt_length_choices))
        batch_size = int(rng.choice(config.profiling.batch_size_choices))
        decode_tokens = max(8, int(config.profiling.decode_tokens + rng.integers(-8, 9)))
        base_cpu_mb = 900.0 + float(rng.normal(0, 30))
        run_id = f"synthetic_{run_index:03d}"
        logger.start_run(
            {
                "run_id": run_id,
                "model_name": config.model.model_name,
                "model_size": config.model.model_size,
                "prompt_id": f"prompt_{run_index:03d}",
                "prompt_length": prompt_length,
                "batch_size": batch_size,
            }
        )

        total_weight_mb = 0.0
        kv_resident_mb = 0.0

        for layer_id in range(config.model.num_layers):
            weight_mb = config.profiling.base_weight_mb + (layer_id * 1.15) + float(rng.uniform(-1.0, 1.0))
            activation_mb = (
                config.profiling.base_activation_mb
                * batch_size
                * math.sqrt(prompt_length / 32.0)
                * float(rng.uniform(0.9, 1.1))
            )
            kv_mb = (
                config.profiling.base_kv_cache_mb_per_token
                * batch_size
                * prompt_length
                * (1.0 + 0.03 * layer_id)
                * float(rng.uniform(0.95, 1.05))
            )
            total_weight_mb += weight_mb
            kv_resident_mb += kv_mb
            gpu_mb = total_weight_mb + kv_resident_mb + activation_mb

            logger.log_tensor_event(
                name=f"layer_{layer_id}.weight",
                tensor_type="weight",
                layer_id=layer_id,
                phase="prefill",
                tensor_shape=[config.model.hidden_size, config.model.hidden_size],
                dtype=config.model.dtype,
                device_before="gpu",
                device_after="gpu",
                latency_ms=0.06 + (prompt_length * 0.0015) + float(rng.uniform(0.0, 0.02)),
                access_count=prompt_length,
                prompt_length=prompt_length,
                batch_size=batch_size,
                step_id=0,
                size_bytes=int(weight_mb * MB),
                memory_used_gpu_mb=gpu_mb,
                memory_used_cpu_mb=base_cpu_mb,
            )
            logger.log_tensor_event(
                name=f"layer_{layer_id}.activation",
                tensor_type="activation",
                layer_id=layer_id,
                phase="prefill",
                tensor_shape=[batch_size, prompt_length, config.model.hidden_size],
                dtype=config.model.dtype,
                device_before="gpu",
                device_after="gpu",
                latency_ms=0.03 + float(rng.uniform(0.0, 0.02)),
                access_count=1,
                prompt_length=prompt_length,
                batch_size=batch_size,
                step_id=0,
                size_bytes=int(activation_mb * MB),
                memory_used_gpu_mb=gpu_mb,
                memory_used_cpu_mb=base_cpu_mb,
            )
            logger.log_tensor_event(
                name=f"layer_{layer_id}.kv_cache",
                tensor_type="kv_cache",
                layer_id=layer_id,
                phase="prefill",
                tensor_shape=[batch_size, prompt_length, config.model.hidden_size],
                dtype=config.model.dtype,
                device_before="gpu",
                device_after="gpu",
                latency_ms=0.02 + (prompt_length * 0.0008) + float(rng.uniform(0.0, 0.02)),
                access_count=prompt_length,
                prompt_length=prompt_length,
                batch_size=batch_size,
                step_id=0,
                size_bytes=int(kv_mb * MB),
                memory_used_gpu_mb=gpu_mb,
                memory_used_cpu_mb=base_cpu_mb,
            )

        logger.log_memory_snapshot(stage="prefill", step_id=0, gpu_mb=total_weight_mb + kv_resident_mb, cpu_mb=base_cpu_mb)

        for decode_step in range(1, decode_tokens + 1):
            active_length = prompt_length + decode_step
            kv_resident_mb = 0.0
            activation_step_mb = 0.0
            for layer_id in range(config.model.num_layers):
                weight_mb = config.profiling.base_weight_mb + (layer_id * 1.15)
                kv_mb = (
                    config.profiling.base_kv_cache_mb_per_token
                    * batch_size
                    * active_length
                    * (1.0 + 0.03 * layer_id)
                    * float(rng.uniform(0.95, 1.05))
                )
                activation_mb = (
                    config.profiling.base_activation_mb
                    * batch_size
                    * float(rng.uniform(0.35, 0.55))
                )
                kv_resident_mb += kv_mb
                activation_step_mb += activation_mb
                gpu_mb = total_weight_mb + kv_resident_mb + activation_step_mb

                logger.log_tensor_event(
                    name=f"layer_{layer_id}.weight",
                    tensor_type="weight",
                    layer_id=layer_id,
                    phase="decode",
                    tensor_shape=[config.model.hidden_size, config.model.hidden_size],
                    dtype=config.model.dtype,
                    device_before="gpu",
                    device_after="gpu",
                    latency_ms=0.05 + float(rng.uniform(0.0, 0.02)),
                    access_count=1,
                    prompt_length=prompt_length,
                    batch_size=batch_size,
                    step_id=decode_step,
                    size_bytes=int(weight_mb * MB),
                    memory_used_gpu_mb=gpu_mb,
                    memory_used_cpu_mb=base_cpu_mb,
                )
                logger.log_tensor_event(
                    name=f"layer_{layer_id}.activation",
                    tensor_type="activation",
                    layer_id=layer_id,
                    phase="decode",
                    tensor_shape=[batch_size, 1, config.model.hidden_size],
                    dtype=config.model.dtype,
                    device_before="gpu",
                    device_after="gpu",
                    latency_ms=0.02 + float(rng.uniform(0.0, 0.01)),
                    access_count=1,
                    prompt_length=prompt_length,
                    batch_size=batch_size,
                    step_id=decode_step,
                    size_bytes=int(activation_mb * MB),
                    memory_used_gpu_mb=gpu_mb,
                    memory_used_cpu_mb=base_cpu_mb,
                )
                logger.log_tensor_event(
                    name=f"layer_{layer_id}.kv_cache",
                    tensor_type="kv_cache",
                    layer_id=layer_id,
                    phase="decode",
                    tensor_shape=[batch_size, active_length, config.model.hidden_size],
                    dtype=config.model.dtype,
                    device_before="gpu",
                    device_after="gpu",
                    latency_ms=0.03 + (active_length * 0.0004) + float(rng.uniform(0.0, 0.02)),
                    access_count=2,
                    prompt_length=prompt_length,
                    batch_size=batch_size,
                    step_id=decode_step,
                    size_bytes=int(kv_mb * MB),
                    memory_used_gpu_mb=gpu_mb,
                    memory_used_cpu_mb=base_cpu_mb,
                )

            logger.log_memory_snapshot(
                stage="decode",
                step_id=decode_step,
                gpu_mb=total_weight_mb + kv_resident_mb + activation_step_mb,
                cpu_mb=base_cpu_mb,
            )

        logger.end_run(status="completed")

    return logger.events_frame(), logger.snapshots_frame(), logger.runs_frame()


def save_synthetic_trace_bundle(
    config: ProjectConfig,
    *,
    num_runs: int | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    events_df, snapshots_df, runs_df = generate_synthetic_traces(config, num_runs=num_runs)
    trace_dir = Path(output_dir) if output_dir else (
        config.paths.data_traces / f"{config.profiling.synthetic_trace_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    trace_dir.mkdir(parents=True, exist_ok=True)
    events_path = trace_dir / "events.csv"
    snapshots_path = trace_dir / "snapshots.csv"
    runs_path = trace_dir / "runs.csv"
    events_df.to_csv(events_path, index=False)
    snapshots_df.to_csv(snapshots_path, index=False)
    runs_df.to_csv(runs_path, index=False)
    config.save(trace_dir / "config.json")
    return {
        "trace_dir": trace_dir,
        "events_path": events_path,
        "snapshots_path": snapshots_path,
        "runs_path": runs_path,
    }


def simulate_trace(
    events_df: pd.DataFrame,
    placements_df: pd.DataFrame,
    *,
    memory_budget_mb: float,
    transfer_cost_per_mb_ms: float,
    decode_transfer_multiplier: float,
    kv_cache_transfer_multiplier: float,
    cpu_resident_penalty_ms: float,
    oom_penalty_ms_per_mb: float,
) -> dict[str, float | int | str | bool]:
    merged = events_df.merge(
        placements_df[["run_id", "tensor_name", "placement", "policy_name"]],
        on=["run_id", "tensor_name"],
        how="left",
    )
    merged["placement"] = merged["placement"].fillna("cpu")
    merged["policy_name"] = merged["policy_name"].fillna("unknown")
    merged["size_mb"] = merged["size_bytes"] / MB

    phase_multiplier = np.where(merged["phase"] == "decode", decode_transfer_multiplier, 1.0)
    type_multiplier = np.where(
        merged["tensor_type"] == "kv_cache",
        kv_cache_transfer_multiplier,
        1.0,
    )
    access_multiplier = 1.0 + np.log1p(merged["access_count"].clip(lower=1)) * 0.10
    transfer_cost_ms = np.where(
        merged["placement"] == "cpu",
        merged["size_mb"] * transfer_cost_per_mb_ms * phase_multiplier * type_multiplier * access_multiplier,
        0.0,
    )
    residency_penalty_ms = np.where(merged["placement"] == "cpu", cpu_resident_penalty_ms, 0.0)
    merged["effective_latency_ms"] = merged["latency_ms"] + transfer_cost_ms + residency_penalty_ms

    step_resident = (
        merged[merged["placement"] == "gpu"]
        .groupby(["run_id", "step_id"], as_index=False)
        .agg(step_gpu_mb=("size_mb", "sum"))
    )
    peak_gpu_mb = float(step_resident["step_gpu_mb"].max()) if not step_resident.empty else 0.0
    budget_violation_mb = max(0.0, peak_gpu_mb - memory_budget_mb)
    budget_penalty_ms = budget_violation_mb * oom_penalty_ms_per_mb

    total_latency_ms = float(merged["effective_latency_ms"].sum() + budget_penalty_ms)
    decode_token_count = (
        merged.loc[merged["phase"] == "decode", ["run_id", "step_id"]]
        .drop_duplicates()
        .shape[0]
    )
    throughput_tps = 0.0 if total_latency_ms <= 0 else (decode_token_count / (total_latency_ms / 1000.0))

    ttft_by_run = (
        merged.loc[(merged["phase"] == "prefill") | ((merged["phase"] == "decode") & (merged["step_id"] == 1))]
        .groupby("run_id", as_index=False)
        .agg(ttft_ms=("effective_latency_ms", "sum"))
    )
    mean_ttft_ms = float(ttft_by_run["ttft_ms"].mean()) if not ttft_by_run.empty else 0.0

    metrics = {
        "policy_name": str(placements_df["policy_name"].iloc[0]) if not placements_df.empty else "unknown",
        "memory_budget_mb": float(memory_budget_mb),
        "total_latency_ms": total_latency_ms,
        "mean_latency_ms": float(merged["effective_latency_ms"].mean()),
        "p95_event_latency_ms": float(np.percentile(merged["effective_latency_ms"], 95)),
        "ttft_ms": mean_ttft_ms,
        "throughput_tps": throughput_tps,
        "peak_gpu_mb": peak_gpu_mb,
        "transfer_mb": float(merged.loc[merged["placement"] == "cpu", "size_mb"].sum()),
        "average_transfer_cost_ms": float(np.mean(transfer_cost_ms)),
        "gpu_placement_count": int((placements_df["placement"] == "gpu").sum()),
        "cpu_placement_count": int((placements_df["placement"] == "cpu").sum()),
        "budget_violation_mb": budget_violation_mb,
        "oom": bool(budget_violation_mb > 0.0),
    }
    return metrics
