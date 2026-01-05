# FACT AI Group 23 - Reproducibility Project

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains the reproducibility project for UvA course **5204FACT6Y (FACT AI)**. Our group reproduces results from a selected research paper on fairness, accountability, confidentiality, or transparency in AI systems.

## Project Structure

```
fact-ai-group-23/
├── src/                    # Source code for model implementation
├── configs/                # Configuration files
├── data/                   # Dataset storage (raw and processed)
├── notebooks/              # Jupyter notebooks for experiments and analysis
├── results/                # Output files, trained models, and evaluation results
├── docs/                   # Documentation
│   ├── DEVIATIONS.md      # Documented deviations from original paper
│   ├── BASELINES.md       # Baseline implementations and results
│   ├── FINDINGS.md        # Key findings and observations
│   └── REPRODUCTION.md    # Step-by-step reproduction instructions
├── requirements.txt        # Python dependencies
├── config.yaml            # Main configuration file
├── train.py               # Training script
├── evaluate.py            # Evaluation script
└── README.md              # This file
```

## Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster training

### Installation

1. Clone the repository:
```bash
git clone https://github.com/giacomograzia/fact-ai-group-23.git
cd fact-ai-group-23
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

Run the training script with default configurations:
```bash
python train.py
```

Or specify a custom configuration file:
```bash
python train.py --config configs/custom_config.yaml
```

### Evaluation

Evaluate a trained model:
```bash
python evaluate.py --model results/model_checkpoint.pth
```

### Running Experiments

Jupyter notebooks for experiments and analysis are available in the `notebooks/` directory:
```bash
jupyter notebook notebooks/
```

## Original Paper

**[Paper Title]**: [Title of the paper being reproduced]

**Authors**: [Authors]

**Published in**: [Conference/Journal, Year]

**Link**: [URL to paper]

**Abstract**: [Brief abstract or summary]

## Reproducibility

This project aims to reproduce the key results from the original paper. For detailed reproduction instructions, see [docs/REPRODUCTION.md](docs/REPRODUCTION.md).

### Key Results

| Metric | Original Paper | Our Reproduction | Difference |
|--------|---------------|------------------|------------|
| [Metric 1] | [Value] | [Value] | [±Value] |
| [Metric 2] | [Value] | [Value] | [±Value] |

## Deviations

Any deviations from the original paper (e.g., implementation details, hyperparameters, data preprocessing) are documented in [docs/DEVIATIONS.md](docs/DEVIATIONS.md).

## Findings

Our key findings and observations are documented in [docs/FINDINGS.md](docs/FINDINGS.md).

## Team Members

- [Member 1 Name] - [Student Number]
- [Member 2 Name] - [Student Number]
- [Member 3 Name] - [Student Number]
- [Member 4 Name] - [Student Number]

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- University of Amsterdam (UvA) - FACT AI Course (5204FACT6Y)
- Original paper authors
- Course instructors and teaching assistants

## Contact

For questions or issues, please open an issue on this repository or contact the team members.