from __future__ import annotations

from pathlib import Path
import json

import pandas as pd


def load_events(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(path))


def load_snapshots(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(path))


def load_runs(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(path))


def load_trace_bundle(trace_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trace_path = Path(trace_dir)
    events = load_events(trace_path / "events.csv")
    snapshots = load_snapshots(trace_path / "snapshots.csv")
    runs = load_runs(trace_path / "runs.csv")
    return events, snapshots, runs


def find_latest_trace_dir(trace_root: str | Path) -> Path:
    root = Path(trace_root)
    candidates = [path for path in root.iterdir() if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No trace directories found under {root}.")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def find_latest_file(root: str | Path, pattern: str) -> Path:
    candidates = list(Path(root).glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No files matching {pattern} under {root}.")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_prompts(path: str | Path) -> list[str]:
    target = Path(path)
    if target.suffix == ".json":
        payload = json.loads(target.read_text(encoding="utf-8"))
        return [str(item) for item in payload]
    return [line.strip() for line in target.read_text(encoding="utf-8").splitlines() if line.strip()]
