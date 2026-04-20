from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence
import json
import time

import pandas as pd
import psutil

try:
    import torch
except ImportError:  # pragma: no cover - torch is a runtime dependency.
    torch = None


class TensorLogger:
    """Collects tensor events and memory snapshots across one or more runs."""

    def __init__(self) -> None:
        self.current_run_id: str | None = None
        self.current_metadata: dict[str, Any] = {}
        self.events: list[dict[str, Any]] = []
        self.snapshots: list[dict[str, Any]] = []
        self.runs: list[dict[str, Any]] = []
        self._run_start_ts: float | None = None

    def start_run(self, metadata: dict[str, Any]) -> str:
        run_id = str(metadata.get("run_id") or f"run_{int(time.time() * 1000)}")
        self.current_run_id = run_id
        self.current_metadata = dict(metadata)
        self.current_metadata["run_id"] = run_id
        self._run_start_ts = time.time()
        return run_id

    def log_tensor_event(
        self,
        *,
        name: str,
        tensor_type: str,
        layer_id: int,
        phase: str,
        tensor: Any | None = None,
        tensor_shape: Sequence[int] | None = None,
        dtype: str | None = None,
        device_before: str = "gpu",
        device_after: str = "gpu",
        latency_ms: float = 0.0,
        access_count: int = 1,
        prompt_length: int | None = None,
        batch_size: int | None = None,
        step_id: int = 0,
        size_bytes: int | None = None,
        memory_used_gpu_mb: float | None = None,
        memory_used_cpu_mb: float | None = None,
        notes: str = "",
    ) -> None:
        if self.current_run_id is None:
            raise RuntimeError("Call start_run before logging tensor events.")

        resolved_shape = self._infer_shape(tensor=tensor, tensor_shape=tensor_shape)
        resolved_size_bytes = self._infer_size_bytes(
            tensor=tensor,
            tensor_shape=resolved_shape,
            dtype=dtype,
            size_bytes=size_bytes,
        )
        resolved_dtype = self._infer_dtype(tensor=tensor, dtype=dtype)
        event = {
            "run_id": self.current_run_id,
            "timestamp": time.time(),
            "model_name": self.current_metadata.get("model_name"),
            "model_size": self.current_metadata.get("model_size"),
            "prompt_id": self.current_metadata.get("prompt_id"),
            "batch_size": batch_size if batch_size is not None else self.current_metadata.get("batch_size"),
            "phase": phase,
            "step_id": step_id,
            "layer_id": layer_id,
            "tensor_name": name,
            "tensor_type": tensor_type,
            "tensor_shape": json.dumps(list(resolved_shape)) if resolved_shape else "[]",
            "dtype": resolved_dtype,
            "device_before": device_before,
            "device_after": device_after,
            "size_bytes": resolved_size_bytes,
            "size_mb": resolved_size_bytes / (1024 ** 2),
            "access_count": access_count,
            "latency_ms": latency_ms,
            "memory_used_gpu_mb": self._current_gpu_mb() if memory_used_gpu_mb is None else memory_used_gpu_mb,
            "memory_used_cpu_mb": self._current_cpu_mb() if memory_used_cpu_mb is None else memory_used_cpu_mb,
            "prompt_length": prompt_length if prompt_length is not None else self.current_metadata.get("prompt_length"),
            "notes": notes,
        }
        self.events.append(event)

    def log_memory_snapshot(
        self,
        *,
        stage: str,
        step_id: int = 0,
        gpu_mb: float | None = None,
        cpu_mb: float | None = None,
        notes: str = "",
    ) -> None:
        if self.current_run_id is None:
            raise RuntimeError("Call start_run before logging snapshots.")
        snapshot = {
            "run_id": self.current_run_id,
            "timestamp": time.time(),
            "stage": stage,
            "step_id": step_id,
            "gpu_mb": self._current_gpu_mb() if gpu_mb is None else gpu_mb,
            "cpu_mb": self._current_cpu_mb() if cpu_mb is None else cpu_mb,
            "notes": notes,
        }
        self.snapshots.append(snapshot)

    def end_run(self, status: str = "completed", notes: str = "") -> dict[str, Any]:
        if self.current_run_id is None:
            raise RuntimeError("Call start_run before ending a run.")
        duration_ms = 0.0
        if self._run_start_ts is not None:
            duration_ms = (time.time() - self._run_start_ts) * 1000.0
        summary = {
            "run_id": self.current_run_id,
            "status": status,
            "notes": notes,
            "duration_ms": duration_ms,
            "event_count": sum(1 for event in self.events if event["run_id"] == self.current_run_id),
            "snapshot_count": sum(1 for snap in self.snapshots if snap["run_id"] == self.current_run_id),
        }
        summary.update(self.current_metadata)
        self.runs.append(summary)
        self.current_run_id = None
        self.current_metadata = {}
        self._run_start_ts = None
        return summary

    def events_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.events)

    def snapshots_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.snapshots)

    def runs_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.runs)

    def save_csv(
        self,
        events_path: str | Path,
        snapshots_path: str | Path | None = None,
        runs_path: str | Path | None = None,
    ) -> tuple[Path, Path | None, Path | None]:
        events_target = Path(events_path)
        events_target.parent.mkdir(parents=True, exist_ok=True)
        self.events_frame().to_csv(events_target, index=False)

        snapshot_target = Path(snapshots_path) if snapshots_path else None
        if snapshot_target is not None:
            snapshot_target.parent.mkdir(parents=True, exist_ok=True)
            self.snapshots_frame().to_csv(snapshot_target, index=False)

        run_target = Path(runs_path) if runs_path else None
        if run_target is not None:
            run_target.parent.mkdir(parents=True, exist_ok=True)
            self.runs_frame().to_csv(run_target, index=False)

        return events_target, snapshot_target, run_target

    def save_json(
        self,
        events_path: str | Path,
        snapshots_path: str | Path | None = None,
        runs_path: str | Path | None = None,
    ) -> tuple[Path, Path | None, Path | None]:
        events_target = Path(events_path)
        events_target.parent.mkdir(parents=True, exist_ok=True)
        events_target.write_text(json.dumps(self.events, indent=2), encoding="utf-8")

        snapshot_target = Path(snapshots_path) if snapshots_path else None
        if snapshot_target is not None:
            snapshot_target.parent.mkdir(parents=True, exist_ok=True)
            snapshot_target.write_text(json.dumps(self.snapshots, indent=2), encoding="utf-8")

        run_target = Path(runs_path) if runs_path else None
        if run_target is not None:
            run_target.parent.mkdir(parents=True, exist_ok=True)
            run_target.write_text(json.dumps(self.runs, indent=2), encoding="utf-8")

        return events_target, snapshot_target, run_target

    @staticmethod
    def _infer_shape(tensor: Any | None, tensor_shape: Sequence[int] | None) -> list[int]:
        if tensor_shape is not None:
            return [int(dim) for dim in tensor_shape]
        if tensor is not None and hasattr(tensor, "shape"):
            return [int(dim) for dim in tensor.shape]
        return []

    @staticmethod
    def _infer_dtype(tensor: Any | None, dtype: str | None) -> str:
        if dtype is not None:
            return dtype
        if tensor is not None and hasattr(tensor, "dtype"):
            return str(tensor.dtype)
        return "unknown"

    @staticmethod
    def _infer_size_bytes(
        tensor: Any | None,
        tensor_shape: Iterable[int],
        dtype: str | None,
        size_bytes: int | None,
    ) -> int:
        if size_bytes is not None:
            return int(size_bytes)
        if tensor is not None and hasattr(tensor, "numel") and hasattr(tensor, "element_size"):
            return int(tensor.numel() * tensor.element_size())
        shape = list(tensor_shape)
        if not shape:
            return 0
        numel = 1
        for dim in shape:
            numel *= int(dim)
        bytes_per_elem = 2 if dtype in {"float16", "torch.float16", "bfloat16", "torch.bfloat16"} else 4
        return int(numel * bytes_per_elem)

    @staticmethod
    def _current_gpu_mb() -> float:
        if torch is None or not torch.cuda.is_available():
            return 0.0
        return float(torch.cuda.memory_allocated() / (1024 ** 2))

    @staticmethod
    def _current_cpu_mb() -> float:
        return float(psutil.Process().memory_info().rss / (1024 ** 2))
