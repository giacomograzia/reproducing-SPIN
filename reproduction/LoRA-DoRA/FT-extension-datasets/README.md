# FT-extension-datasets

This directory contains Slurm job scripts and output logs for running the **LoRA/DoRA fine-tuning pipeline on extension datasets** (safety + helpfulness) across multiple base models. The pipeline is **Train → Merge → Eval** using LlamaFactory for training/merging and the project’s evaluation scripts for metrics.

## Contents

- `jobs/full_pipeline_dora_lora_all_models.job`
  - Runs LoRA and DoRA fine-tuning for **Qwen2-7B-Instruct** and **Mistral-7B-Instruct-v0.2**, then merges adapters and evaluates.
- `jobs/full_pipeline_lora_dora_vicuna.job`
  - Runs LoRA and DoRA fine-tuning for **Vicuna-7B-v1.5**, then merges adapters and evaluates.
- `jobs/lora_dora_eval.job`
  - Standalone evaluation job for a merged Vicuna checkpoint (safety + helpfulness).
- `outputs/`
  - Stored stdout/err logs and metrics outputs from prior runs.

## Prerequisites

These job scripts assume:

- A Slurm cluster with an H100 GPU partition (`#SBATCH --partition=gpu_h100`).
- The project is located at `~/fact-ai-group-23`.
- LlamaFactory is present at `~/fact-ai-group-23/LlamaFactory`.
- Conda environments named `Llama` (for training/merging) and `FACT` (for evaluation).
- Datasets registered in `LlamaFactory/data/dataset_info.json`:
  - `safety`
  - `helpfulness_train`
- A writable scratch folder (default: `/scratch-shared/$USER`).

## How to run

Submit the job file you need using Slurm:

- Qwen + Mistral:
  - `sbatch jobs/full_pipeline_dora_lora_all_models.job`
- Vicuna:
  - `sbatch jobs/full_pipeline_lora_dora_vicuna.job`
- Eval only (edit `MODEL_PATH` inside the job first):
  - `sbatch jobs/lora_dora_eval.job`

### Optional dry-run

To only create output directories and skip heavy work:

- `DRY_RUN=1 sbatch jobs/full_pipeline_dora_lora_all_models.job`
- `DRY_RUN=1 sbatch jobs/full_pipeline_lora_dora_vicuna.job`

## Outputs

Each full pipeline job writes artifacts under scratch:

- Adapters: `$SCRATCH_SHARED/fact-ai-group-23/extension/runs/llamafactory_saves_safe_help/adapters/`
- Merged models: `$SCRATCH_SHARED/fact-ai-group-23/extension/runs/llamafactory_saves_safe_help/merged/`
- Metrics: `$SCRATCH_SHARED/fact-ai-group-23/extension/runs/llamafactory_saves_safe_help/metrics/`

Logs are written to the paths defined in each job file’s `#SBATCH --output` / `#SBATCH --error`.

## Notes

- The pipeline uses **LoRA** for both methods; **DoRA** is enabled with `--use_dora True`.
- Helpfulness evaluation uses `src/eval_run.py` with `data/helpfulness_test.csv`; safety evaluation uses `src/eval_safety.py` with `--category safety`.
- If a run fails early, check the Slurm `.err` file first; the scripts include traps that print the failing command.
