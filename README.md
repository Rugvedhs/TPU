# Learned Memory Placement for LLM Serving

This repository is a small research scaffold for learned memory placement in LLM inference. The goal is to measure which tensors are hot, which ones are expensive to move, and whether a learned placement policy can beat simple baselines under tight GPU memory budgets.

The first version is intentionally small:

- a tensor trace logger
- a synthetic profiling workload that produces realistic-enough traces without a production serving stack
- rule-based baselines
- a small PyTorch MLP that learns GPU vs CPU placement decisions from traces
- a benchmark and plotting pipeline that produces tables and figures for a paper

## What This Code Answers

The project is structured to answer concrete systems questions:

- Which tensors are largest?
- Which tensors are hottest?
- How does KV-cache grow over time?
- When does GPU memory become the bottleneck?
- How much does CPU offload hurt latency and throughput?
- Does a learned placement policy beat all-GPU, naive overflow, and a hand-written heuristic?

## Repository Layout

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
├── src/
├── scripts/
├── data/
├── results/
└── paper/
```

Core implementation lives in `src/`, command-line entry points live in `scripts/`, generated traces go to `data/traces/`, and figures/tables go to `results/`.

## Quick Start

1. Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

2. Generate synthetic tensor traces:

```bash
python scripts/run_profile.py --num-runs 24
```

3. Train the learned placement policy:

```bash
python scripts/run_train.py
```

4. Evaluate baselines vs the learned policy:

```bash
python scripts/run_eval.py
```

## Outputs

The starter workflow writes:

- trace logs with tensor metadata and memory snapshots
- summary tables such as hottest tensors and memory by layer
- policy training history with loss and validation reward
- benchmark tables for throughput, latency, transfer cost, and budget violations
- figures for KV-cache growth, access frequency by layer, and method-vs-budget comparisons

## Important Notes

- The default profiler is synthetic on purpose. It gives you fast iteration and lets you debug the learning and benchmarking pipeline before integrating a real model runner.
- The logger and data schema are designed so you can swap in a real inference trace later without rewriting the rest of the project.
- The benchmark uses an explicit cost model for CPU-resident tensors and transfer penalties. That is a research scaffold, not a production memory manager.

## Suggested Next Steps

After the starter pipeline works end to end, the next upgrades are:

- replace the synthetic profiler with a hooked Hugging Face causal LM
- add a real KV-cache trace during prefill and decode
- plug policy decisions into an actual serving loop
- compare across models and memory budgets
- export paper-ready figures and claims
