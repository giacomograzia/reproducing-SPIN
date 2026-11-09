# N=1 Localization Controls

Two experiments extending the existing SPIN reproduction (`src/*.py`), testing the robustness of SPIN's strong N=1 (single-example) localization result.

## Purpose

**Experiment 1 — Relevant vs Unrelated N=1.** Tests whether SPIN's strong performance with a single (N=1) fairness/privacy localization example depends on that example's *semantic relevance*. 25 paired runs: Condition A (relevant) localizes from one real fairness example + one real privacy example; Condition B (unrelated) localizes from two length-matched, unrelated Alpaca examples instead. Same seed/run-ID schedule pairs run *i* across both conditions. Paired permutation tests + bootstrap CIs (fairness, privacy), Holm-corrected.

**Experiment 2 — N=1 Localization Stability.** Tests whether N=1 localization recovers weight sets similar to a full-data (N=1000, the entire fairness/privacy dataset) reference localization, via Jaccard similarity of the final coupled (general-capability-filtered) weight-index sets, and whether that stability correlates (Spearman) with downstream fairness/privacy gains across the 25 relevant N=1 runs.

Both experiments reuse the existing SPIN pipeline unchanged: `src/compute_importance_score.py` (Importance Score localization, weight-level, `score = |W| * |grad_W L|`), `src/SPIN.py` (general-capability-filtered coupled-set suppression), `src/eval_salad.py` (fairness/privacy awareness evaluation via MD-Judge). See "Reused modules" below.

## Results

All three registered models (Qwen2-7B-Instruct, Mistral-7B-Instruct-v0.2, Vicuna-7B-v1.5) have completed the full pipeline: 150 GPU runs total (25 relevant + 25 unrelated + 1 N=1000 reference, ×3 models). Raw outputs in `results/<model>/`, computed stats in `analysis/<model>/`, figures in `figures/<model>/` and `figures/grouped_figures/`.

### Experiment 1 — Relevant vs Unrelated N=1

| Model | Metric | Relevant | Unrelated | Diff | Permutation p | Holm p | 95% CI | Significant? |
|---|---|---|---|---|---|---|---|---|
| Qwen2-7B | Fairness | 0.7379±0.0240 | 0.7138±0.0132 | +0.0240 | 0.0006 | 0.0006 | [+0.0128, +0.0354] | ✅ |
| | Privacy | 0.8173±0.0252 | 0.7878±0.0191 | +0.0295 | 0.0002 | 0.0004 | [+0.0162, +0.0418] | ✅ |
| Mistral-7B | Fairness | 0.7090±0.0058 | 0.6996±0.0048 | +0.0094 | 0.0002 | 0.0004 | [+0.0065, +0.0123] | ✅ |
| | Privacy | 0.7483±0.0117 | 0.7347±0.0094 | +0.0135 | 0.0014 | 0.0014 | [+0.0068, +0.0203] | ✅ |
| Vicuna-7B | Fairness | 0.6490±0.0353 | 0.6454±0.0281 | +0.0036 | 0.6477 | 0.6477 | [−0.0121, +0.0186] | ❌ |
| | Privacy | 0.5093±0.0478 | 0.4868±0.0438 | +0.0225 | 0.0562 | 0.1124 | [+0.0005, +0.0439] | ❌ (marginal) |

**Qwen2 and Mistral replicate the core finding cleanly** — relevant beats unrelated significantly on both metrics, all 4 Holm-adjusted p-values under 0.002, all 4 CIs clear of zero. **Vicuna does not replicate**: fairness is essentially flat, and privacy is only marginal before correction and non-significant after Holm. This is a genuine cross-model difference (all three use n=25), not a sample-size artifact. Vicuna's origin (unsuppressed) baseline is also notably weaker/less-validated than the other two (single-run provenance, privacy origin=0.4658 vs. Qwen's 0.767 and Mistral's 0.717 — see `configs/models.yaml`), a plausible contributing factor worth investigating further.

### Experiment 2 — N=1 Localization Stability

| Model | Final-coupled Jaccard (mean±std) | Jaccard vs Fairness gain | Jaccard vs Privacy gain |
|---|---|---|---|
| Qwen2-7B | 0.0782±0.0138 | rho=0.206, Holm p=0.653 | rho=−0.124, Holm p=0.653 |
| Mistral-7B | 0.0707±0.0148 | rho=0.015, Holm p=0.931 | rho=0.225, Holm p=0.553 |
| Vicuna-7B | 0.0620±0.0337 | rho=0.100, Holm p=1.000 | rho=0.080, Holm p=1.000 |

Consistent across all three models: N=1 localizations overlap only modestly (~6–8%) with the full-data reference, and **none of the six correlations (2 metrics × 3 models) reach significance**. This part of the finding replicates uniformly — no model shows N=1 localization converging toward the full-data weight set, and stability doesn't predict downstream performance anywhere.

### Interpretation

1. **"N=1 works and is content-sensitive" is model-dependent, not universal** — confirmed on 2 of 3 tested models (Qwen2, Mistral), not on Vicuna.
2. **"Not explained by weight-recovery" is robust across all three models** — a clean 3-for-3 replication, arguably the more universal finding of the two.
3. Vicuna is a documented exception, not noise, and is a natural target for follow-up investigation (e.g. whether its weaker/unvalidated origin baseline, or some architecture-specific behavior, explains the null result).

## Environment

Conda env `FACT` (see `../../environment.yml`). No new dependency in practice: `scipy` (used for all statistics — permutation tests, bootstrap CIs, Spearman correlation) is already installed in the `FACT` env as a transitive dependency of `outlines` (a vLLM dependency), confirmed via `pip show scipy` / `pip check`, and now listed explicitly in `environment.yml` for documentation.

Models: this experiment supports multiple models via a registry at `configs/models.yaml`. Currently registered: `qwen2-7b-instruct` (Qwen2-7B-Instruct, matching the existing low-data sweep), `vicuna-7b-v1.5` (lmsys/vicuna-7b-v1.5), `mistral-7b-instruct-v0.2` (mistralai/Mistral-7B-Instruct-v0.2). Every script takes a required `--model <label>` argument resolved against this registry; see "Adding a new model" below.

**Compute policy**: every step below is submitted via `sbatch`, never run inline on the login node — including CPU-only steps (manifest building, analysis) — per the project compute policy.

**Storage policy**: every job's `--score_dir`/`--model_dir` resolves under `$TMPDIR` (node-local scratch, auto-cleaned at job end), never `$HOME`. Only small artifacts (`masks.pt` flat-index tensors, `run_result.json`, the one-row slot CSVs) live under `results/` in this repo tree, per this experiment's own requirement that all outputs stay inside `n1_localization_controls/`.

## Directory structure

Everything model-specific (manifests, Alpaca pool cache, run results, analysis outputs, figures, SLURM logs) is nested one level deeper by model label. `n1_experiment_config.yaml` (shared hyperparameters) and `models.yaml` (the model registry) stay flat, since neither varies by model.

```
n1_localization_controls/
├── README.md                     -- this file
├── PLAN.md                       -- original experiment specification
├── configs/
│   ├── n1_experiment_config.yaml -- shared hyperparameters (no model info)
│   ├── models.yaml                -- model registry: label -> HF path + origin baseline provenance
│   ├── qwen2-7b-instruct/
│   │   ├── n1_run_manifest.csv/json            -- generated: the 50-row (25 runs x 2 conditions) manifest
│   │   └── alpaca_candidate_pool_token_lengths.csv -- generated: cached Alpaca pool tokenization
│   ├── vicuna-7b-v1.5/{...}                    -- same shape, per model
│   └── mistral-7b-instruct-v0.2/{...}
├── scripts/
│   ├── model_registry.py         -- shared helper: load/lookup configs/models.yaml
│   ├── build_n1_manifest.py      -- Experiment 1 sampling (both conditions), --model <label>
│   ├── spin_artifacts.py         -- library: SPIN suppression + artifact capture (model-agnostic)
│   ├── run_n1_run.py             -- per-run driver (localize, suppress, evaluate), --model <label>
│   ├── build_reference_localization.py -- Experiment 2 Step 1 (N=1000 reference), --model <label>
│   ├── jobs/                     -- SLURM job templates (one per pipeline step), MODEL_LABEL-parameterized
│   └── slurm_outputs/<model>/    -- SLURM .out logs, one subdir per model
├── results/<model>/
│   ├── relevant_n1/run_00..24/   -- Condition A outputs
│   ├── unrelated_n1/run_00..24/  -- Condition B outputs (+ generated slot CSVs)
│   └── reference/                -- N=1000 reference localization
├── analysis/                     -- stats scripts (model-agnostic) + generated CSV/JSON summaries under <model>/
└── figures/                      -- plot scripts (model-agnostic) + generated PNG/PDF figures under <model>/
    └── grouped_figures/          -- cross-model comparison figures (1xN, one panel per model)
```

## Commands (run in this order, all via `sbatch` from `src/`)

Every job defaults to `MODEL_LABEL=qwen2-7b-instruct` if not overridden, so the bare commands below reproduce the original Qwen2 run. For any other registered model, pass `--export=ALL,MODEL_LABEL=<label>` **and** an explicit `--output` override (SLURM `#SBATCH --output=` is fixed at submission time and can't read `$MODEL_LABEL`) — see "Adding a new model" below for the full form.

```bash
cd ~/fact-ai-group-23/src

# 1. Build the 25-run manifest + unrelated-condition slot CSVs
sbatch n1_localization_controls/scripts/jobs/build_manifest.job

# 2. Build the N=1000 full-data reference localization (Experiment 2 Step 1)
sbatch n1_localization_controls/scripts/jobs/reference_localization.job

# 3. Run the 25 relevant N=1 runs (array job, one task per run)
sbatch n1_localization_controls/scripts/jobs/relevant_n1_array.job

# 4. Run the 25 unrelated N=1 runs (array job, one task per run)
sbatch n1_localization_controls/scripts/jobs/unrelated_n1_array.job

# 5. Run the full analysis + figure pipeline (after 2-4 complete)
sbatch n1_localization_controls/scripts/jobs/run_analysis.job

# 6. Once step 5 has completed for two or more models, generate cross-model
#    comparison figures (1xN, one panel per model) -- by default includes
#    every model registered in configs/models.yaml, skipping any with no
#    analysis output yet (with a warning, not silently):
sbatch n1_localization_controls/scripts/jobs/run_grouped_analysis.job
```

Step 5 internally runs, in order (each with `--model <label>`):
```bash
python n1_localization_controls/analysis/build_summary_tables.py --model <label>
python n1_localization_controls/analysis/stats_experiment1.py --model <label>
python n1_localization_controls/analysis/compute_jaccard_stability.py --model <label>
python n1_localization_controls/analysis/stats_stability_vs_performance.py --model <label>
python n1_localization_controls/analysis/plot_experiment1.py --model <label>
python n1_localization_controls/analysis/plot_experiment2_stability.py --model <label>
```

Before trusting `analysis/stats_utils.py` on real data, its smoke test (tiny hand-computable examples) should pass:
```bash
python n1_localization_controls/analysis/test_stats_utils.py
```
(Genuinely brief, sub-second — the one script in this experiment that may run directly rather than via `sbatch`.)

## Expected output files

All paths below are relative to `n1_localization_controls/`, with `<model>` being the label passed to that step (e.g. `qwen2-7b-instruct`):

| Step | Output |
|---|---|
| 1 | `configs/<model>/n1_run_manifest.{csv,json}`, `configs/<model>/alpaca_candidate_pool_token_lengths.csv`, `results/<model>/unrelated_n1/run_XX/{fairness,privacy}_slot.csv` |
| 2 | `results/<model>/reference/masks.pt`, `results/<model>/reference/reference_result.json` |
| 3 | `results/<model>/relevant_n1/run_XX/{run_result.json,masks.pt,run.log}` (25x) |
| 4 | `results/<model>/unrelated_n1/run_XX/{run_result.json,masks.pt,run.log}` (25x) |
| 5 | `analysis/<model>/results_{relevant,unrelated}_n1_summary.{csv,json}`, `analysis/<model>/experiment1_paired.{csv,json}`, `analysis/<model>/experiment1_stats.{csv,json}`, `analysis/<model>/jaccard_stability_results.{csv,json}`, `analysis/<model>/stability_vs_performance_stats.{csv,json}`, `figures/<model>/experiment1_paired_relevant_vs_unrelated.{png,pdf}`, `figures/<model>/experiment2_jaccard_vs_fairness_privacy_delta.{png,pdf}` |
| 6 | `figures/grouped_figures/experiment1_{fairness,privacy}_grouped.{png,pdf}`, `figures/grouped_figures/experiment2_{fairness,privacy}_grouped.{png,pdf}` |

## Seed / run-ID pairing

`seed = run_id` for both conditions (0–24) — the same schedule is reused across Condition A and B, so `results/relevant_n1/run_i` and `results/unrelated_n1/run_i` are a matched pair for all paired statistics in Experiment 1. The manifest (`configs/n1_run_manifest.json`) is the single source of truth for which example(s) each `(run_id, condition)` uses; every downstream script reads it (or its run-level outputs) rather than resampling.

## Resuming interrupted experiments

Every run script (`run_n1_run.py`, `build_reference_localization.py`) checks its own `run_result.json`/`reference_result.json` for `"complete": true` at start and exits immediately if already done. Resubmitting any of the `sbatch` jobs above (including the array jobs) is always safe — completed runs are skipped, only missing/incomplete ones recompute. Pass `--force` to a script directly to force a recompute of one specific run.

## Adding a new model

1. **Register it** in `configs/models.yaml`: add a `<label>` entry with `path` (HF repo id), `origin_fairness`, `origin_privacy`, `origin_source`, `origin_decoding`, `origin_source_script`, `origin_note` (provenance for the unsuppressed-model baseline — reuse an existing origin run if one exists elsewhere in the repo, per the greedy-decoding/seed-invariance justification already used for Qwen2/Vicuna/Mistral; only compute a fresh one if genuinely none exists).
2. **Build the manifest**: `mkdir -p scripts/slurm_outputs/<label>` then
   ```bash
   sbatch --export=ALL,MODEL_LABEL=<label> \
     --output=$HOME/fact-ai-group-23/src/n1_localization_controls/scripts/slurm_outputs/<label>/N1_Build_Manifest_%A.out \
     n1_localization_controls/scripts/jobs/build_manifest.job
   ```
3. **Build the reference localization** (same `--export`/`--output` pattern, `reference_localization.job`).
4. **Smoke test — mandatory before the full sweep.** Submit one single run (e.g. `sbatch --export=ALL,MODEL_LABEL=<label> --array=0-0 --output=... relevant_n1_array.job`) and check its `run_result.json` for `"complete": true` with sane (non-NaN) awareness rates. The `gpu_memory_utilization=0.55` / MD-Judge `torch_dtype=torch.float16` A100 settings in `run_n1_run.py` were only empirically confirmed for Qwen2-7B + MD-Judge-v0.1 (see the project GPU memory policy) — do not assume they transfer silently to a new model without this check, even though same-parameter-class models are expected to work.
5. **Full sweep**: submit `relevant_n1_array.job` and `unrelated_n1_array.job` with the same `--export`/`--output` pattern (50 runs total).
6. **Analysis**: submit `run_analysis.job` with the same `--export`/`--output` pattern — writes `analysis/<label>/` and `figures/<label>/`.

No `.job` file, Python script, or directory needs editing for any of this — only step 1 (a `models.yaml` entry) and the `--export`/`--output` flags on each `sbatch` call.

## Known deviations from "keep everything identical to the existing low-data experiment"

1. `spin_artifacts.py` duplicates `SPIN.py`'s per-layer orchestration loop (using its `find_layers`/`get_set_difference_mask` unmodified) since that function returns nothing and offers no artifact-capture hook.
2. `run_n1_run.py` duplicates `main.py`'s save-model/vLLM/MD-Judge tail (every function it calls is imported unmodified) since `main.py` has no importable pipeline entry point.
3. `eval_ppl` (wikitext perplexity) is skipped for all 51 new runs — not required by this experiment's run-level outputs.
4. The unsuppressed-model baseline for each registered model is reused from an existing origin run in the repo (see `configs/models.yaml`'s per-model `origin_source`) rather than rerun per-run, justified by greedy/deterministic decoding making it seed-invariant.
5. The unrelated-example CSV generation + token-length-matching infrastructure (`build_n1_manifest.py`) has no repo precedent — required because `get_alpaca()` ignores the `--dataset` label for its 3 recognized aliases, and SPIN's suppression step needs the save/load labels to match exactly per slot.
6. Statistics use `scipy.stats` (already installed transitively, see Environment) rather than a hand-rolled implementation.
7. Jaccard computation requires composite `(layer, module, flat_index)` keys — SPIN's own code never compares indices across layers/modules, so this is a new correctness rule specific to this analysis.
8. The Alpaca token-length candidate pool is bounded to 5,000 (of 45,874) rows to bound CPU tokenization time — a config constant, easily raised.
9. General/Alpaca N=128 importance scores are recomputed independently for each of the 50 runs (not cached/shared across conditions) to avoid race conditions between the two concurrently-running SLURM arrays.

Full technical justification for each of these is in the approved implementation plan (kept alongside this experiment's development history).

## Terminology notes

- All localization is **weight-level** (flat indices into full `[out_features, in_features]` `nn.Linear` weight matrices), matching SPIN's own Importance Score definition — never "neuron"-level.
- The N=1000 localization is referred to as the **"full-data reference localization"** or **"large-N reference"** — never "ground truth": SPIN does not establish it as the objectively correct fairness/privacy-coupled weight set, it is simply the largest-data localization available for comparison.
- `raw_coupled_idx` (the "coupled set before general-capability filtering") is a **new diagnostic artifact defined for this analysis only** — it is not an object SPIN's own code computes or names. `final_coupled_idx` is, by construction, byte-for-byte identical to SPIN's own `filtered_indices` (the exact weights it zeros).
