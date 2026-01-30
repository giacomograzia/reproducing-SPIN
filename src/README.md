# src

This folder contains the main experiment scripts, evaluation entry points, and utilities for the FACT-AI project.

## Structure

- [src/main.py](src/main.py): primary entry point that wires together model loading, data selection, and evaluation flow.
- [src/utils.py](src/utils.py): shared helpers (I/O, formatting, seeding, and common utilities referenced by multiple scripts).
- [src/SPIN.py](src/SPIN.py): SPIN-related methods and helpers used by experiments.
- [src/compute_importance_score.py](src/compute_importance_score.py): computes neuron/feature importance scores used by analysis and reporting.
- [src/eval_origin.py](src/eval_origin.py): runs origin/faithfulness evaluation and writes metrics for reporting.
- [src/eval_helpfulness.py](src/eval_helpfulness.py): loads prompt/response CSVs, generates model answers with vLLM, and scores with Prometheus-2 rubrics (used by extension helpfulness evals).
- [src/eval_safety.py](src/eval_safety.py): runs safety/fairness/privacy evaluation with MD-Judge v0.1 over vLLM outputs (used by extension safety evals).
- [src/eval_salad.py](src/eval_salad.py): runs SALAD evaluation metrics.
- [src/eval_ppl.py](src/eval_ppl.py): computes perplexity-based evaluation metrics.
- [src/eval_run.py](src/eval_run.py): CLI driver that wires dataset loading, vLLM generation, and Prometheus-2 judging for helpfulness (extension eval entry point).
- [src/data/](src/data/): datasets and cached data artifacts consumed by eval scripts.
- [src/jobs/](src/jobs/): job submission scripts and batch utilities for running experiments on the cluster.
- [src/new-neuron-locating-method/](src/new-neuron-locating-method/): experimental implementations for neuron-locating methods.
- [src/results output/](src/results%20output/): generated results and outputs (metrics, logs, and intermediate artifacts).

## Notes

- For reproduction scripts and end-to-end pipelines, see [reproduction/](reproduction/).
