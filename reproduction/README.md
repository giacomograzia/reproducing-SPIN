# Reproducing Results Instructions for SFT Methods

This folder contains scripts and documentation to reproduce SFT results for FFT and LoRA/DoRA variants.

## Reproduction folder quick overview

- [reproduction/README.md](reproduction/README.md): top-level instructions and common setup.
- [reproduction/FFT/](reproduction/FFT/): full fine-tuning (FFT) assets and scripts.
- [reproduction/LoRA-DoRA/](reproduction/LoRA-DoRA/): LoRA/DoRA assets and scripts.
  - [reproduction/LoRA-DoRA/FT-safe-datasets/README.md](reproduction/LoRA-DoRA/FT-safe-datasets/README.md): safe dataset pipeline and usage.
  - [reproduction/LoRA-DoRA/FT-unsafe-datasets/README.md](reproduction/LoRA-DoRA/FT-unsafe-datasets/README.md): unsafe dataset pipeline and usage.
- [reproduction/general_capabilities/](reproduction/general_capabilities/): general capability evaluation assets.
- [reproduction/artifacts/](reproduction/artifacts/): stored outputs and plots.

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

Note: All values are from the original paper (Appendix F).

- Epochs: 3
- Optimizer: AdamW with $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 1 \times 10^{-8}$, weight decay $0$.
- Scheduler: cosine decay with warmup ratio $0.1$.
