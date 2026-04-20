from __future__ import annotations

from pathlib import Path
import argparse
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import build_default_config, ensure_directories, set_global_seed
from src.data_loader import find_latest_trace_dir, load_events
from src.train import train_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the learned placement policy from trace logs.")
    parser.add_argument("--events", type=str, default=None, help="Path to an events.csv file. Defaults to the latest trace.")
    parser.add_argument("--budget-mb", type=float, default=None, help="GPU memory budget used for oracle labels.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional output directory for model artifacts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = build_default_config(ROOT, seed=args.seed)
    ensure_directories(config)
    set_global_seed(config.seed)

    events_path = Path(args.events) if args.events else find_latest_trace_dir(config.paths.data_traces) / "events.csv"
    events_df = load_events(events_path)
    artifacts = train_policy(
        config,
        events_df,
        output_dir=args.output_dir,
        memory_budget_mb=args.budget_mb,
    )
    print(f"Training data: {events_path}")
    print(f"Model artifact: {artifacts['model_path']}")
    print(f"History CSV: {artifacts['history_path']}")
    print(f"Decision examples: {artifacts['decisions_path']}")


if __name__ == "__main__":
    main()
