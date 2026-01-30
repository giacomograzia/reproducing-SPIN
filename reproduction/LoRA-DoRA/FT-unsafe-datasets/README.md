# FT-unsafe-datasets

This directory contains Slurm job scripts and output metrics for running the **LoRA/DoRA fine-tuning pipeline on unsafe datasets** (privacy + fairness) for multiple base models. The pipeline is **Train → Merge → Eval** using LlamaFactory for training/merging and the project’s evaluation scripts for metrics.

## Contents

- `full_pipeline_lora_dora_unsafe_QWEN_MISTRAL.job`
  - Runs LoRA and DoRA fine-tuning for **Qwen2-7B-Instruct** and **Mistral-7B-Instruct-v0.2**, then merges adapters and evaluates.
- `full_pipeline_lora_dora_unsafe_VICUNA.job`
  - Runs LoRA and DoRA fine-tuning for **Vicuna-7B-v1.5**, then merges adapters and evaluates.
- `metrics-mistral/`, `metrics-qwen/`, `metrics-vicuna/`
  - Stored evaluation outputs from prior runs (stdout/err + metrics artifacts).

## Prerequisites

These job scripts assume:

- A Slurm cluster with an H100 GPU partition (`#SBATCH --partition=gpu_h100`).
- The project is located at `~/fact-ai-group-23`.
- LlamaFactory is present at `~/fact-ai-group-23/LlamaFactory`.
- Conda environments named `LlFactory` (for training/merging) and `FACT` (for evaluation).
- Datasets registered in `LlamaFactory/data/dataset_info.json`:
  - `beaver_train330k_privacy_unsafe_1k`
  - `beaver_train330k_fairness_unsafe_1k`
- A writable scratch folder (default: `/scratch-shared/$USER`).

## How to run

Submit the job file you need using Slurm:

- Qwen + Mistral:
  - `sbatch full_pipeline_lora_dora_unsafe_QWEN_MISTRAL.job`
- Vicuna:
  - `sbatch full_pipeline_lora_dora_unsafe_VICUNA.job`

### Optional dry-run

To only create output directories and skip heavy work:

- `DRY_RUN=1 sbatch full_pipeline_lora_dora_unsafe_QWEN_MISTRAL.job`
- `DRY_RUN=1 sbatch full_pipeline_lora_dora_unsafe_VICUNA.job`

## Outputs

Each job writes artifacts under scratch:

- Adapters: `$SCRATCH_SHARED/fact-ai-group-23-runs/llamafactory_saves_*/adapters/`
- Merged models: `$SCRATCH_SHARED/fact-ai-group-23-runs/llamafactory_saves_*/merged/`
- Metrics: `$SCRATCH_SHARED/fact-ai-group-23-runs/llamafactory_saves_*/metrics/`

Logs are written to the paths defined in each job file’s `#SBATCH --output` / `#SBATCH --error`.

## Notes

- The pipeline uses **LoRA** for both methods; **DoRA** is enabled with `--use_dora True`.
- Hyperparameters (epochs, LR, batch size, etc.) are set according to the paper by Qian et al., 2025 (Appendix F) inside each job file.
- If a run fails early, check the Slurm `.err` file first; each script has a trap that prints the failing command.
