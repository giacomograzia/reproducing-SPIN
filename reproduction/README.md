# Reproducing SFT Results

This folder contains scripts and documentation to reproduce SFT results for FFT and LoRA/DoRA variants.

## Reproduction folder quick overview

- [FFT/](FFT/): full fine-tuning (FFT) assets and scripts.
  - [FFT/finetuning/](FFT/finetuning/): main FFT training and evaluation jobs.
  - [FFT/extension/](FFT/extension/): extension experiment jobs.
  - [FFT/limited-data-exp/](FFT/limited-data-exp/): limited-data experiment jobs and logs.
  - [FFT/malicious-exp/](FFT/malicious-exp/): malicious-data experiment jobs and outputs.
- [LoRA-DoRA/](LoRA-DoRA/): LoRA/DoRA fine-tuning assets and scripts.
  - [LoRA-DoRA/FT-safe-datasets/](LoRA-DoRA/FT-safe-datasets/): safe dataset pipelines and metrics.
  - [LoRA-DoRA/FT-unsafe-datasets/](LoRA-DoRA/FT-unsafe-datasets/): unsafe dataset pipelines and metrics.
  - [LoRA-DoRA/FT-extension-datasets/](LoRA-DoRA/FT-extension-datasets/): extension dataset pipelines and outputs.
  - [LoRA-DoRA/FT-data-scale/](LoRA-DoRA/FT-data-scale/): data scaling experiment scripts and documentation.
- [general_capabilities/](general_capabilities/): general capability evaluation assets and results.
- [figures_and_tables/](figures_and_tables/): result collection scripts, stored outputs, tables, and plots.

## Environment setup

```bash
conda create -n LlFactory
conda activate LlFactory

# install LlamaFactory
cd LlamaFactory/
pip install -e .
pip install -r requirements/metrics.txt

# open LlamaFactory web UI
python -m llamafactory.cli webui
```

## Hyperparameters

Note: All values are from the original paper (see Appendix D in Reproducibility Study).

- Epochs: 3
- Optimizer: AdamW with $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 1 \times 10^{-8}$, weight decay $0$.
- Scheduler: cosine decay with warmup ratio $0.1$.
