# Reproduction Instructions

This document provides step-by-step instructions to reproduce the results from the original paper.

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended)
- At least 8GB RAM
- At least 10GB free disk space

## Step 1: Environment Setup

### 1.1 Clone the Repository

```bash
git clone https://github.com/giacomograzia/fact-ai-group-23.git
cd fact-ai-group-23
```

### 1.2 Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 1.3 Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 1.4 Verify Installation

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Step 2: Data Preparation

### 2.1 Download Dataset

[Provide instructions or script to download the dataset]

```bash
# Option 1: Manual download
# Download from [URL] and place in data/raw/

# Option 2: Automated download
python scripts/download_data.py
```

### 2.2 Data Preprocessing

```bash
python scripts/preprocess_data.py --config configs/data_config.yaml
```

**Expected output**:
- Processed training data: `data/processed/train.csv`
- Processed validation data: `data/processed/val.csv`
- Processed test data: `data/processed/test.csv`

### 2.3 Verify Data

```bash
python scripts/verify_data.py
```

## Step 3: Training

### 3.1 Train Base Model

Train the model with default configuration:

```bash
python train.py --config config.yaml
```

**Expected training time**: [X hours on GPU / Y hours on CPU]

**Expected output**:
- Model checkpoints: `results/models/checkpoint_epoch_*.pth`
- Training logs: `results/logs/train.log`
- TensorBoard logs: `results/logs/tensorboard/`

### 3.2 Monitor Training

To monitor training progress with TensorBoard:

```bash
tensorboard --logdir results/logs/tensorboard/
```

Open browser and navigate to `http://localhost:6006`

### 3.3 Train with Custom Configuration

To train with custom hyperparameters:

```bash
python train.py --config configs/custom_config.yaml
```

## Step 4: Evaluation

### 4.1 Evaluate on Test Set

```bash
python evaluate.py --model results/models/final_model.pth --config config.yaml
```

**Expected output**:
- Evaluation results: `results/evaluation_results.txt`
- Detailed metrics: `results/detailed_metrics.json`

### 4.2 Fairness Evaluation

```bash
python evaluate.py --model results/models/final_model.pth \
                   --config config.yaml \
                   --fairness-evaluation
```

### 4.3 Baseline Comparison

```bash
python scripts/run_baselines.py --config config.yaml
```

## Step 5: Analysis and Visualization

### 5.1 Run Jupyter Notebooks

```bash
jupyter notebook notebooks/
```

Key notebooks:
- `01_exploratory_data_analysis.ipynb`: Data exploration
- `02_model_training_analysis.ipynb`: Training analysis
- `03_fairness_evaluation.ipynb`: Fairness metrics analysis
- `04_results_visualization.ipynb`: Results visualization

### 5.2 Generate Plots

```bash
python scripts/generate_plots.py --results results/evaluation_results.txt
```

## Step 6: Reproduce Specific Experiments

### Experiment 1: [Experiment Name]

```bash
python train.py --config configs/experiment1_config.yaml
python evaluate.py --model results/models/experiment1_model.pth
```

**Expected results**: [Describe expected results]

### Experiment 2: [Experiment Name]

```bash
python train.py --config configs/experiment2_config.yaml
python evaluate.py --model results/models/experiment2_model.pth
```

**Expected results**: [Describe expected results]

## Step 7: Verification

### 7.1 Compare with Original Paper

Compare your results with Table X in the original paper:

| Metric | Original Paper | Our Reproduction | Difference |
|--------|---------------|------------------|------------|
| Accuracy | [value] | [value] | [±value] |
| Fairness Metric | [value] | [value] | [±value] |

### 7.2 Statistical Significance

```bash
python scripts/statistical_tests.py --results results/
```

## Troubleshooting

### Common Issues

#### Issue 1: CUDA Out of Memory

**Solution**: Reduce batch size in `config.yaml`:
```yaml
data:
  batch_size: 16  # or 8
```

#### Issue 2: Data Download Fails

**Solution**: [Provide solution]

#### Issue 3: Training Diverges

**Solution**: [Provide solution]

### Getting Help

If you encounter issues not listed here:
1. Check existing GitHub issues
2. Open a new issue with:
   - Error message
   - System configuration
   - Steps to reproduce

## Expected Results Summary

After completing all steps, you should have:

1. ✅ Trained model with accuracy within ±X% of original paper
2. ✅ Fairness metrics within ±Y of original paper
3. ✅ All baseline comparisons completed
4. ✅ Visualization plots generated
5. ✅ Analysis notebooks executed

## Time Estimate

- **Setup**: 30 minutes
- **Data preparation**: 1-2 hours
- **Training**: [X] hours
- **Evaluation**: 30 minutes
- **Analysis**: 1-2 hours

**Total**: [Y] hours

## Additional Resources

- Original paper: [link]
- Dataset documentation: [link]
- Related implementations: [links]

## Citation

If you use this reproduction in your work, please cite:

```bibtex
@misc{fact-ai-group-23,
  title={Reproducibility Study: [Paper Title]},
  author={[Your Names]},
  year={2026},
  url={https://github.com/giacomograzia/fact-ai-group-23}
}
```
