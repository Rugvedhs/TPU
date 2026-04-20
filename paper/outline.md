# Paper Outline

## Problem

LLM serving is increasingly bottlenecked by memory capacity and memory movement rather than raw compute alone.

## Hypothesis

A learned tensor-placement policy can reduce transfer cost and improve throughput under tight GPU memory budgets compared with fixed hand-written rules.

## Method

- collect tensor traces during inference
- build baseline placement policies
- train a small policy network from trace-derived features
- simulate and benchmark placement outcomes under varying memory budgets

## Evaluation

- throughput in tokens/sec
- time to first token
- mean and p95 latency
- peak GPU memory
- transfer volume and transfer cost

## Key Figures

- KV-cache growth over time
- access frequency by layer
- throughput vs memory budget
- latency vs memory budget
- learned policy vs baselines
