from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json
import random

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch is a runtime dependency.
    torch = None


@dataclass
class PathConfig:
    root: Path
    notebooks: Path
    scripts: Path
    src: Path
    data_raw: Path
    data_traces: Path
    data_processed: Path
    results_figures: Path
    results_tables: Path
    results_logs: Path
    paper: Path


@dataclass
class ModelConfig:
    model_name: str = "synthetic-transformer-small"
    model_size: str = "125M"
    num_layers: int = 12
    hidden_size: int = 768
    dtype: str = "float16"


@dataclass
class ProfilingConfig:
    synthetic_runs: int = 24
    prompt_length_choices: tuple[int, ...] = (32, 64, 128, 256, 512)
    batch_size_choices: tuple[int, ...] = (1, 2, 4)
    decode_tokens: int = 48
    base_weight_mb: float = 16.0
    base_activation_mb: float = 0.8
    base_kv_cache_mb_per_token: float = 0.010
    synthetic_trace_name: str = "synthetic_trace"


@dataclass
class TrainingConfig:
    hidden_dim: int = 64
    dropout: float = 0.10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    epochs: int = 30
    validation_fraction: float = 0.20
    train_budget_mb: float = 768.0


@dataclass
class BenchmarkConfig:
    memory_budgets_mb: tuple[float, ...] = (256.0, 384.0, 512.0, 768.0, 1024.0)
    transfer_cost_per_mb_ms: float = 0.45
    decode_transfer_multiplier: float = 1.15
    kv_cache_transfer_multiplier: float = 1.25
    cpu_resident_penalty_ms: float = 0.15
    oom_penalty_ms_per_mb: float = 4.0


@dataclass
class PlotConfig:
    dpi: int = 160
    style: str = "ggplot"


@dataclass
class ProjectConfig:
    seed: int
    paths: PathConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    profiling: ProfilingConfig = field(default_factory=ProfilingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    plot: PlotConfig = field(default_factory=PlotConfig)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["paths"] = {key: str(value) for key, value in payload["paths"].items()}
        return payload

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return target


def build_default_config(root: str | Path | None = None, seed: int = 7) -> ProjectConfig:
    root_path = Path(root or Path(__file__).resolve().parents[1]).resolve()
    paths = PathConfig(
        root=root_path,
        notebooks=root_path / "notebooks",
        scripts=root_path / "scripts",
        src=root_path / "src",
        data_raw=root_path / "data" / "raw",
        data_traces=root_path / "data" / "traces",
        data_processed=root_path / "data" / "processed",
        results_figures=root_path / "results" / "figures",
        results_tables=root_path / "results" / "tables",
        results_logs=root_path / "results" / "logs",
        paper=root_path / "paper",
    )
    return ProjectConfig(seed=seed, paths=paths)


def ensure_directories(config: ProjectConfig) -> None:
    for path in config.paths.__dict__.values():
        Path(path).mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
