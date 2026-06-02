# Safety-Helpfulness Trade-off Datasets

Scripts for assembling the trade-off extension evaluation datasets used by the reproducibility study. Each script pulls a public HF dataset, filters/samples it, and writes local artifacts (HF `save_to_disk` plus CSV/JSON when applicable).

## Structure

- `extension_datasets/helpfulness.py`: builds HelpSteer helpfulness splits by filtering `helpfulness == 4`, sampling a train set, and saving train/test as HF datasets plus CSV/JSON prompt/response files.
- `extension_datasets/safety_train.py`: builds a safe-response training sample from PKU-SafeRLHF-30K by selecting a safe/safer response per prompt, sampling 1,000 rows, and exporting HF + CSV/JSON.
- `extension_datasets/safety_test.py`: builds a toxic-prompt evaluation CSV from Salad-Data (base_set), optionally subsampling, and writing `data/salad_toxic.csv`.

## Notes

- All scripts rely on `datasets.load_dataset` and expect internet access the first time you run them.
- Output folders/files are written relative to the repo root (e.g., `helpsteer_val_h4_prompt_response_split/`, `pku_saferlhf30k_train_safe_safer_prompt_response_sample1000.*`, `data/salad_toxic.csv`).
