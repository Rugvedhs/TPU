# Project Gameplan

This file is the short version of the plan for getting this repo from starter scaffold to a research prototype strong enough to show a professor and eventually package for a workshop submission.

## Goal

Build a credible prototype for learned GPU/CPU memory placement in LLM inference that:

- logs real tensor behavior
- compares against real baselines
- trains a learned placement policy
- produces plots, tables, and a clear paper claim

## Your Part

You only need to do a small number of things.

1. Keep everything in this one repo folder.
2. Put the repo on GitHub.
3. Use Google Colab with a T4 GPU for early runs.
4. Run the install and experiment cells I give you.
5. Share errors or outputs when something breaks.
6. Push local changes to GitHub when you want Colab to see them.
7. If we later need a paid GPU, create the account and handle payment yourself.

## My Part

I can handle almost all of the actual engineering work in this repo.

1. Set up and organize the codebase.
2. Build the real tracing pipeline.
3. Add model loading and inference hooks.
4. Add prompt workloads and data processing.
5. Implement baselines.
6. Implement the learned policy and training loop.
7. Build evaluation, plots, tables, and summaries.
8. Keep the codebase clean and reproducible.
9. Help shape the experiment story so it is worth showing a professor.

## What I Still Cannot Do For You

Even with broad access, there are a few things you still need to handle.

1. Create or log into external accounts like GitHub, Google, Runpod, Lambda, or Hugging Face.
2. Complete payment, 2FA, captchas, or approval screens.
3. Decide if you want to spend money on a paid GPU.

## Immediate Setup

Do these first.

1. Create a GitHub repo for this project.
2. Push this repo to GitHub.
3. Open Google Colab.
4. Start a notebook with a T4 GPU, not a TPU.
5. Clone the GitHub repo in Colab.

## Short-Term Milestones

### Milestone 1: Professor-ready prototype

Target: 5 to 10 days of focused work.

Done means:

- a small real Hugging Face model runs
- tensor traces are real, not only synthetic
- at least 3 baselines run
- the learned policy trains
- the repo produces a few strong figures
- you have something intelligent to show a professor

### Milestone 2: Workshop-style project

Target: 2 to 4 weeks for a credible early workshop prototype
Target: 4 to 8 weeks for something much stronger and cleaner

Done means:

- real experiments over multiple memory budgets
- prompt-length sweeps
- stronger heuristics
- ablations
- clean tables and plots
- a clear claim, even if the result is mixed

## Best Honest Timeline

If you stay responsive and we keep scope tight:

- professor-ready: around 1 week
- early NeurIPS-workshop-level prototype: around 2 to 4 weeks
- much stronger workshop-quality codebase: around 4 to 8 weeks

This does not guarantee acceptance. It does mean we can plausibly get to a serious, showable research project.

## Exact Workflow

1. I edit the repo here.
2. You push the repo to GitHub.
3. Colab pulls the latest repo.
4. You run the notebook cells.
5. You send me errors, outputs, or screenshots.
6. I fix the code and keep moving.

## What To Say To A Professor Soon

Once Milestone 1 is done, your pitch is:

"I'm building a learned memory-placement system for LLM inference that decides what stays in fast GPU memory versus CPU memory under tight budgets. I already have tracing, baselines, a learned policy, and an evaluation pipeline, and now I'm testing whether learned placement beats heuristic offload on real workloads."

## Next Concrete Step

Your next step:

1. Make the GitHub repo.
2. Tell me the repo URL.

My next step after that:

1. I will give you the exact Colab clone and install cells.
2. Then I will upgrade this repo from scaffold to real-model tracing.
