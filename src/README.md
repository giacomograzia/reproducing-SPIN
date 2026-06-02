# src

This folder contains the main experiment scripts, evaluation entry points, and utilities for the reproducibility study.

## Structure

- [main.py](main.py): primary experiment.
- [utils.py](utils.py): shared helpers used by the experiment scripts.
- [SPIN.py](SPIN.py): SPIN-related methods and helpers.
- [compute_importance_score.py](compute_importance_score.py): computes neuron importance scores.
- [eval_origin.py](eval_origin.py): origin-model evaluation.
- [eval_helpfulness.py](eval_helpfulness.py): helpfulness evaluation script.
- [eval_safety.py](eval_safety.py): safety evaluation script.
- [eval_salad.py](eval_salad.py): SALAD evaluation script.
- [eval_ppl.py](eval_ppl.py): perplexity evaluation script.
- [eval_run.py](eval_run.py): evaluation CLI driver.
- [data/](data/): datasets consumed by the (original) experiment and evaluation scripts.
- [jobs/](jobs/): cluster job scripts for SPIN and extension experiments.
- [new-neuron-locating-method/](new-neuron-locating-method/): experimental neuron-locating job and output for extension in reproducibility study.
- [multiseed_logs/](multiseed_logs/): multiseed limited-data evaluation logs and results.
- [output_spin/](output_spin/): stored SPIN experiment outputs.
- [output_extension/](output_extension/): stored extension experiment outputs.

## Notes

- For reproduction scripts and end-to-end pipelines, see [reproduction/](reproduction/).
