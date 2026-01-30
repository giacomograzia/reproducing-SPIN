# Data Scaling Experiments

This directory contains scripts and job files for running data scaling experiments to determine how much fine-tuning data is needed before privacy and fairness awareness drops.

## Overview

The experiment tests:
- **Model**: Qwen2-7B-Instruct
- **Method**: LoRA
- **8 dataset sizes** (per dataset): 600, 400, 200, 100, 75, 50, 25, 10
  - Total samples per size: 1200, 800, 400, 200, 150, 100, 50, 20

**Total runs**: 1 model × 1 method × 8 sizes = **8 experiments**

## Balanced Sampling Strategy

### How LlamaFactory Ensures 50/50 Sampling

The key to balanced sampling is using LlamaFactory's `interleave_probs` parameter:

```bash
--dataset "privacy_dataset,fairness_dataset"
--mix_strategy interleave_under
--interleave_probs "0.5,0.5"
```

**What this does:**
- `interleave_under`: Stops when the smaller dataset is exhausted (ensures equal contribution)
- `interleave_probs "0.5,0.5"`: Each batch has 50% probability of sampling from privacy and 50% from fairness
- With equal-sized datasets (created by our preprocessing), this guarantees exactly 50/50 split

**Alternative approach (used here):**
We pre-create separate datasets with exactly N samples each, then combine them. This is simpler and more transparent than relying solely on probabilistic sampling.

## Files

### 1. `create_subsampled_datasets.py`
Python script that creates balanced subsampled datasets.

**What it does:**
- Loads full safe datasets (1k samples each)
- For each target size N, randomly samples N samples from privacy and N from fairness
- Saves subsampled datasets to `LlamaFactory/data/`
- Updates `LlamaFactory/data/dataset_info.json` with new entries

### 2. `data_scaling_experiments.job`
SLURM job file with nested loops:
```
for model in (qwen2_7b):
  for method in (lora):
    for size in (600, 400, 200, 100, 75, 50, 25, 10):
      train -> merge -> evaluate
```

## Usage

### Step 1: Create Subsampled Datasets

```bash
cd /home/scur0168/fact-ai-group-23/reproduction/LoRA-DoRA
python create_subsampled_datasets.py
```

**Expected output:**
- 16 new JSON files in `LlamaFactory/data/`:
  - `beaver_train330k_privacy_safe_600.json`
  - `beaver_train330k_fairness_safe_600.json`
  - `beaver_train330k_privacy_safe_400.json`
  - ... (and so on for all 8 sizes)
- Updated `LlamaFactory/data/dataset_info.json` with 16 new entries

**Verification:**
```bash
# Check files were created
ls -lh ../LlamaFactory/data/beaver_train330k_*_safe_*.json

# Check dataset_info.json was updated
grep "beaver_train330k_privacy_safe_" ../LlamaFactory/data/dataset_info.json
```

### Step 2: Submit the Experiment Job

```bash
sbatch data_scaling_experiments.job
```

**Estimated runtime**: 
- Each run takes ~20-40 minutes (train + merge + eval)
- Total: 8 runs → **~3–6 hours**

### Step 3: Monitor Progress

```bash
# Check job status
squeue -u $USER

# Monitor logs in real-time
tail -f logs/data_scale_*.out

# Check for errors
tail -f logs/data_scale_*.err
```

## Output Structure

All outputs are saved to `/scratch-shared/$USER/fact-ai-group-23-runs/llamafactory_data_scale/`:

```
llamafactory_data_scale/
├── adapters/
│   ├── qwen2_7b_lora_n600_*/
│   ├── qwen2_7b_lora_n400_*/
│   └── ... (8 adapter directories)
├── merged/
│   ├── qwen2_7b_lora_n600_*_merged/
│   ├── qwen2_7b_lora_n400_*_merged/
│   └── ... (8 merged model directories)
└── metrics/
  ├── qwen2_7b_lora_n600_*_merged/
  │   ├── eval.out
  │   └── eval.err
  └── ... (8 metric directories)
```

## Dataset Naming Convention

Format: `beaver_train330k_{category}_safe_{size}`

Examples:
- `beaver_train330k_privacy_safe_600`: 600 privacy safe samples
- `beaver_train330k_fairness_safe_100`: 100 fairness safe samples

When combined: `privacy_safe_600,fairness_safe_600` = 1200 total samples (50/50 split)

## Hyperparameters

All experiments use the same hyperparameters (from paper):
- **Epochs**: 3
- **Learning rate**: 3e-4
- **Batch size**: 32
- **Optimizer**: AdamW (β1=0.9, β2=0.999, ε=1e-8, wd=0)
- **LR scheduler**: Cosine with 0.1 warmup ratio
- **LoRA rank**: 8
- **LoRA alpha**: 16
- **Precision**: BF16
