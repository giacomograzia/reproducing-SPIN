# Quick Start Guide: Data Scaling Experiments (LoRA)

## Summary

Minimal setup to test how LoRA performs with decreasing amounts of fine-tuning data. Ensures **balanced 50/50 sampling** from fairness and privacy datasets.

## What I Created

### 1. **create_subsampled_datasets.py**
Generates balanced subsampled datasets:
- **Input**: Full datasets (1k samples each)
- **Output**: 16 new datasets (8 sizes × 2 categories)
- **Sizes**: 600, 400, 200, 100, 75, 50, 25, 10 samples per category
- **Auto-updates**: `dataset_info.json` with new entries (`LlamaFactory/data/dataset_info.json`)

### 2. **data_scaling_experiments.job**
SLURM job with nested loops:
- **Loop 1**: Model (Qwen2-7B)
- **Loop 2**: Method (LoRA)
- **Loop 3**: Dataset sizes (8 sizes)
- **Total**: 8 experiments (1 × 1 × 8)

### 3. **README_data_scaling.md**
Short doc on balanced sampling and usage.

## Quick Start Commands

```bash
# Step 1: Create subsampled datasets (takes ~30 seconds)
cd /home/scur0168/fact-ai-group-23/reproduction/LoRA-DoRA_v2
python create_subsampled_datasets.py

# Step 2: Submit experiments (runs ~3–6 hours total)
sbatch data_scaling_experiments.job
```

## How 50/50 Balanced Sampling Works

### Method 1: Pre-processing (What we use)
**Most reliable approach**
- Create separate files with exactly N samples from each category
- Combine during training: `privacy_safe_N,fairness_safe_N`
- Result: Exactly N from privacy + N from fairness = 2N total

### Method 2: LlamaFactory's `interleave_probs`
Used during training for balance:
```bash
--dataset "privacy_safe_600,fairness_safe_600"
--mix_strategy interleave_under
--interleave_probs "0.5,0.5"
```

## File Locations

```
/home/scur0168/fact-ai-group-23/
├── reproduction/LoRA-DoRA/
│   ├── create_subsampled_datasets.py  ← Run this first
│   ├── data_scaling_experiments.job   ← SLURM job
│   ├── README_data_scaling.md         ← Short doc
│   └── logs/                          ← Job output logs
│
└── LlamaFactory/
   └── data/
      ├── beaver_train330k_privacy_safe_600.json   ← Created by script
      ├── beaver_train330k_fairness_safe_600.json  ← Created by script
      ├── ... (14 more files)
      └── dataset_info.json                        ← Auto-updated

/scratch-shared/$USER/fact-ai-group-23-runs/
└── llamafactory_data_scale/
   ├── adapters/     ← 8 trained adapters
   ├── merged/       ← 8 merged models
   └── metrics/      ← 8 evaluation results
```