from __future__ import annotations

from pathlib import Path
import argparse
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import build_default_config, ensure_directories, set_global_seed
from src.simulator import save_synthetic_trace_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic tensor traces for memory-placement experiments.")
    parser.add_argument("--num-runs", type=int, default=None, help="Number of synthetic runs to generate.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional explicit output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = build_default_config(ROOT, seed=args.seed)
    ensure_directories(config)
    set_global_seed(config.seed)

    artifacts = save_synthetic_trace_bundle(
        config,
        num_runs=args.num_runs,
        output_dir=args.output_dir,
    )
    print(f"Trace bundle: {artifacts['trace_dir']}")
    print(f"Events CSV: {artifacts['events_path']}")
    print(f"Snapshots CSV: {artifacts['snapshots_path']}")
    print(f"Runs CSV: {artifacts['runs_path']}")


if __name__ == "__main__":
    main()
