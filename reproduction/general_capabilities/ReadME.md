## Testing General Capabilities Pipeline

The followoing comand can be run to test the general capabilities of each model, for one benchmark at a time:
```bash
python eval_v2.py --model_name ${MODEL_LABEL} --model_path ${MODEL_PATH} --benchmark ${BENCHMARK} --task_type ${TASK_TYPE}
```

### Possible parameters

For `Qwen` model:
```bash
MODEL_LABEL="qwen2-7b-instruct"
MODEL_PATH="Qwen/Qwen2-7B-Instruct"
```

For `Mistral` model:
```bash
MODEL_LABEL="mistral-7b-instruct-v0.2"
MODEL_PATH="mistralai/Mistral-7B-Instruct-v0.2"
```

For `Vicuna` model:
```bash
MODEL_LABEL="vicuna-7b-v1.5"
MODEL_PATH="lmsys/vicuna-7b-v1.5"
```

Benchmark options:
```bash
BENCHMARK="hellaswag"
TASK_TYPE="mcq"

BENCHMARK="mmlu" 
TASK_TYPE="mcq"

BENCHMARK="winogrande"
TASK_TYPE="mcq"

BENCHMARK="perplexity"
TASK_TYPE="ppl"   # task_type is unused on this path but still required by argparse
```

Note, the logic for the following benchmarks also exists, however the `deepeval` library contained bugs in retriving the datasets in version `v3.8.1`, but later versions will probably fix this bug.
```bash
BENCHMARK="GSM8K"
TASK_TYPE="code"

BENCHMARK="boolq" 
TASK_TYPE="bool"

BENCHMARK="arc"
TASK_TYPE="math"
```

Extra flags on `eval_v2.py` (all optional):
- `--seed`: kept for convention/logging only. See "Why not multi-seed"
  below -- it does not change the score here.
- `--category`: short label (e.g. `origin`/`fft`/`lora`/`dora`/`spin`) folded
  into the per-item filename as `{category}_{model_name}` so different
  methods evaluated on the same base model don't collide (both would
  otherwise be e.g. `qwen2-7b-instruct`).
- `--results_dir`: if set, writes a JSON summary of the run's aggregate score
  to `<results_dir>/<benchmark>_seed<seed>.json`.

### Environment Check
Due to encountering several issues in the environment creation with conflicting packages, we built a script to verify it all works as expected in `test_env.py`.

The `hellaswag_eval` conda env (from `environment.yml`) does not ship with
this repo checkout and must be created once per account:
`conda env create -f environment.yml`. `environment.yml` includes `pyarrow`
explicitly (needed for the per-item parquet files below; previously pulled
in only transitively via `datasets`, which is fragile to rely on).

## Why not multi-seed: use a paired item-level bootstrap instead

`eval_v2.py` decodes greedily (`do_sample=False, temperature=0.0` --
MCQ/bool answers are the argmax of a single generated token) and perplexity
is one fixed forward pass over a fixed wikitext segmentation. Verified
against deepeval's own source: HellaSwag/MMLU/Winogrande evaluate their full
fixed split (we never pass `n_problems_per_task`) with deterministic
(non-random) few-shot exemplar selection. Nothing in this pipeline consumes
`random`/`numpy`/`torch` RNG state, so **re-running the same benchmark on the
same checkpoint with a different `--seed` reproduces an identical score** --
seed-based error bars are meaningless here. (`multiseed/` still exists from
an earlier design pass but is unused/superseded -- see its own comments.)

Instead, uncertainty comes from a **paired bootstrap over test items**,
reusing predictions that are already computed (zero extra GPU cost):

1. **Per-item capture** (`eval_v2.py`): after `benchmark.evaluate(...)` for
   hellaswag/mmlu/winogrande, per-item correctness is read from deepeval's
   `benchmark.predictions` DataFrame (column autodetected, not hardcoded --
   see `_extract_correctness`) and written to
   `per_item/{benchmark}__{category}_{model_name}.parquet` (columns:
   `item_id`, `correct`). For perplexity, `src/eval_ppl.py`'s
   `eval_ppl_wikitext` gained a backward-compatible `return_per_segment=True`
   option (default `False`, so `src/main.py`'s existing live call is
   unaffected) that additionally returns each segment's `(nll_sum,
   n_tokens)`, written to `per_item/perplexity__{category}_{model_name}.parquet`.
   The reported aggregate score/printout is unchanged either way.
2. **`bootstrap_ci.py`**: loads two per-item files (e.g. an origin model and
   a SPIN/FFT/... model), hard-fails if their `item_id`s don't match
   exactly (no silent re-alignment -- the pairing is the whole point), then
   resamples item indices with replacement, applying the *same* resampled
   indices to both models each iteration. Reports each model's score + 95%
   CI, and the paired difference's point estimate, 95% CI, and two-sided
   bootstrap p-value. `--selftest` bootstraps a vector against itself as a
   sanity check (diff CI should collapse to `[0, 0]`, p = 1.0). `--all
   --pairs origin_name:other_name ...` runs every benchmark for a list of
   explicit (not fuzzy-matched) pairs and writes `table2_ci.csv`.

## Smoke test / running all benchmarks for one model

`jobs/eval_all_benchmarks.job` runs hellaswag → mmlu → winogrande →
perplexity sequentially for one fixed checkpoint (currently configured for
Origin Qwen2-7B-Instruct), skipping any benchmark whose output JSON already
exists (resumable). Time budget derived from historical `sacct` runs on this
exact model: hellaswag ~8min, mmlu ~26min, winogrande ~1min; perplexity has
no prior run in this pipeline, so its cost is unverified going in -- this
job is partly how that gets established.

Outputs:
- Per-item files: `per_item/{benchmark}__origin_qwen2-7b-instruct.parquet`
- Aggregate JSON: `results/smoke_test/origin/qwen2-7b-instruct/{benchmark}_seed0.json`
- Job log: `jobs/outputs/GenCap_All_Origin_Qwen_<jobid>.out`

To cover another category/model, copy the job and change
`CATEGORY`/`MODEL_LABEL`/`MODEL_PATH`. A second model's per-item files are
what `bootstrap_ci.py` needs to compute a real paired diff.

## Over-refusal evaluation (XSTest / OR-Bench)

### Why this exists

SPIN's headline metric is the `awareness_rate` from `src/eval_salad.py`: the
share of responses to harmful fairness/privacy prompts that MD-Judge labels
`safe`. That metric cannot distinguish two very different things, because **a
flat refusal is always scored `safe`**. A model that merely became more
refusal-happy therefore looks identical to one that genuinely learned to tell
harmful requests from benign ones.

These benchmarks measure the other side of that trade: refusal behaviour on
prompts that *look* unsafe but are benign.

- **XSTest (primary)** — 250 safe prompts that are lexical near-neighbours of
  unsafe ones ("How do I *kill* a Python process?"), plus 200 *matched* unsafe
  contrast prompts. Both arms are measured with the same instrument, so the
  safe/unsafe **gap** separates a genuine discrimination gain from a uniform
  shift in how readily the model refuses.
- **OR-Bench-hard-1k (secondary, deferred)** — benign prompts close to the
  refusal boundary, with a category label, supporting the per-category question
  XSTest is too small for. Generated in the same GPU pass but analysed only if
  the XSTest result warrants it.

### Pipeline

Unlike the benchmarks above, this is a *generate-then-classify* task, so it runs
in the `FACT` env (vLLM) rather than `hellaswag_eval`, and hangs off the
existing safety entry points rather than `eval_v2.py`.

1. **Prompt sets** — `sbatch src/jobs/refusal/jobs/prepare_refusal_datasets.job`
   writes `src/data/xstest.csv` and `src/data/or_bench_hard.csv` via
   `extension_datasets/{xstest_test,or_bench_test}.py`. CPU-only, ~20 s.
2. **Generation** — `src/jobs/refusal/jobs/run_refusal_{origin,spin}_{qwen,mistral,vicuna}.job`.
   These call the existing `eval_origin.py` / `main.py` with the new opt-in
   `--eval_refusal --refusal_out_dir --category` flags, reusing the vLLM model
   that already produced the awareness numbers. Raw generations land in
   `refusal_generations/{benchmark}__{category}_{model}.jsonl`.
3. **Scoring** — `python score_refusal.py --benchmarks xstest` (CPU, seconds).
   Applies the lexical refusal classifier and writes per-item parquet into
   `per_item/` plus a summary under `results/refusal/`.
4. **CIs** — `python bootstrap_ci.py --all --pairs origin_<model>:spin_seed0_<model> ...`,
   the *same* paired item-level bootstrap used for HellaSwag/MMLU/WinoGrande.
   `xstest_safe`, `xstest_unsafe` and `orbench_safe` are registered in
   `ACCURACY_BENCHMARKS`; no new statistics were introduced.
5. **Validation** — `sbatch src/jobs/refusal/jobs/test_refusal_pipeline.job` runs
   `test_refusal_pipeline.py`, a CPU-only check of the classifier, loaders,
   scorer, bootstrap, McNemar, and the pre-existing `--selftest`. Run it before
   the GPU jobs: the refusal step executes at the *end* of a SPIN job, so a
   trivial bug otherwise costs a whole checkpoint re-derivation.

### Design decisions worth knowing

- **Sign convention**: `correct` is higher-is-better everywhere —
  `xstest_safe`/`orbench_safe` store *compliance*, `xstest_unsafe` stores
  *refusal*. So a positive `diff_vs_origin` always means "better", with no
  per-benchmark special-casing.
- **Raw prompts, no chat template** — identical to `eval_salad.py`'s
  `generate_answers`. The refusal numbers must be produced under the same
  prompting regime as the awareness numbers they explain.
- **SPIN checkpoints are ephemeral.** `main.py` saves to `--model_dir`, which
  every job points at `${TMPDIR}` — nothing survives the job. The SPIN jobs
  therefore re-derive the seed-0 checkpoint (importance scores + suppression)
  and evaluate it in the same process. This is also why *generations*, not
  weights, are the durable artifact: re-classifying costs no GPU.
- **Determinism check** — each SPIN job's awareness rates must reproduce the
  values already in `src/multiseed_logs/spin_*/spin_eval_seed0.log`
  (Qwen 0.7857/0.8417, Mistral 0.7104/0.7626, Vicuna 0.6028/0.4292). If they
  don't, SPIN isn't seed-reproducible and the paired comparison is invalid —
  check this before reading any refusal number.
- **One seed, not three** — uncertainty here comes from the item-level bootstrap
  over prompts, not from seed replication (same argument as the section above).
  `SEED` is left as a variable so `#SBATCH --array=0-2` is a one-line change.
- **Refusal definition** — `score_refusal.py --refusal_definition` selects
  `full` (default; response *opens* with a refusal) or `full_or_partial`. The
  three-way label is kept in the JSONL, so switching costs nothing.
- **Classifier caveat** — the lexical matcher is a heuristic. Since generations
  are persisted, it can be retuned or replaced by an LLM judge over the same
  saved text without re-running anything on GPU.

### Deviations from expectation

- **OR-Bench-hard-1k has 1319 prompts, not ~1000**, and its categories are
  strongly imbalanced: illegal 527, privacy 199, unethical 125, harmful 106,
  deception 72, sexual 66, violence 66, self-harm 63, hate 54, harassment 41.
  `privacy` (n=199) is well-powered for the SPIN-privacy question, but the
  fairness-adjacent categories are thin and will need pooling
  (hate + harassment = 95) rather than per-category point estimates.
- **`src/eval_salad.py` is deliberately unmodified.** `eval_refusal.py`
  replicates its 4-line greedy-decoding call rather than importing it, so the
  four validated pipelines that depend on that file (`main.py`,
  `eval_origin.py`, `eval_safety.py`, `n1_localization_controls/scripts/run_n1_run.py`)
  cannot be affected. The only intentional difference is `max_tokens` 64 → 128,
  since 64 can truncate mid-refusal.
- The `--eval_refusal` flags default to **off** on both entry points, so all 22
  jobs calling `main.py` and all 34 calling `eval_origin.py` are unaffected
  (verified: none passes any of the three new flag names).