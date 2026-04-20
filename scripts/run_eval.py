from __future__ import annotations

from pathlib import Path
import argparse
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import build_default_config, ensure_directories, set_global_seed
from src.data_loader import find_latest_file, find_latest_trace_dir, load_events
from src.evaluate import evaluate_project


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark baselines and the learned placement policy.")
    parser.add_argument("--events", type=str, default=None, help="Path to an events.csv file. Defaults to the latest trace.")
    parser.add_argument("--policy-path", type=str, default=None, help="Path to a saved .pt policy. Defaults to the latest one.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = build_default_config(ROOT, seed=args.seed)
    ensure_directories(config)
    set_global_seed(config.seed)

    events_path = Path(args.events) if args.events else find_latest_trace_dir(config.paths.data_traces) / "events.csv"
    policy_path = Path(args.policy_path) if args.policy_path else find_latest_file(config.paths.results_logs, "*.pt")

    events_df = load_events(events_path)
    outputs = evaluate_project(config, events_df, policy_path=policy_path)

    print(f"Benchmark input: {events_path}")
    print(f"Policy: {policy_path}")
    for label, path in outputs.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
