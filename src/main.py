from __future__ import annotations

from src.config import build_default_config, ensure_directories, set_global_seed
from src.data_loader import load_events
from src.evaluate import evaluate_project
from src.simulator import save_synthetic_trace_bundle
from src.train import train_policy


def run() -> None:
    config = build_default_config()
    ensure_directories(config)
    set_global_seed(config.seed)

    trace_artifacts = save_synthetic_trace_bundle(config)
    events_df = load_events(trace_artifacts["events_path"])
    training_artifacts = train_policy(config, events_df)
    evaluate_project(config, events_df, policy_path=training_artifacts["model_path"])


if __name__ == "__main__":
    run()
