from datasets import load_dataset
import random
import json
import csv
import re
import os

# Same PKU-SafeRLHF-30K safety-awareness construction as safety_train.py, but
# split into a genuine train/test pair (grouped by prompt, like helpfulness.py
# does for HelpSteer) instead of only producing a training set. safety_train.py
# has no test counterpart at all -- pairing it with salad_safety.csv as an
# external "held-out" set turned out to leak: 47 of salad_safety.csv's 3246
# prompts already appear in the PKU-SafeRLHF 1k training sample. Splitting by
# prompt within the SAME source, with a leakage assert, avoids that instead of
# relying on two different datasets happening not to overlap.


def clean_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


# Load PKU-SafeRLHF and keep examples with at least one safe response.
ds = load_dataset("PKU-Alignment/PKU-SafeRLHF-30K", split="train")
ds = ds.filter(lambda x: x["is_response_0_safe"] or x["is_response_1_safe"])


# Choose the safest response available per example -- identical selection
# logic to safety_train.py (unique safe response, or safer_response_id when
# both are safe).
def pick_response(ex):
    s0 = ex["is_response_0_safe"]
    s1 = ex["is_response_1_safe"]

    if s0 and s1:
        chosen = int(ex["safer_response_id"])
    elif s0:
        chosen = 0
    else:
        chosen = 1

    return {
        "prompt": ex["prompt"],
        "response": ex[f"response_{chosen}"],
    }


ds_pr = ds.map(pick_response, remove_columns=ds.column_names)

# Train sample size and RNG seed -- same values as safety_train.py, so the
# training half of this split reproduces the paper's original 1000-sample
# construction; the remainder becomes the test set (same pattern as
# helpfulness.py: sample N_TRAIN prompts for train, everything else is test).
N_TRAIN = 1000
SEED = 42

# Group by PROMPT, not row -- PKU-SafeRLHF has multiple independently
# annotated response-pairs for some prompts (this is exactly why
# safety_train.py's row-level sample has 80 duplicate prompts out of 1000).
# Splitting by prompt group instead of row index guarantees no prompt crosses
# the train/test boundary. Only the FIRST row per prompt is kept in each
# split (rather than every response-pair row for that prompt), so train comes
# out to exactly N_TRAIN rows -- matching the paper's literal "1000 samples"
# -- instead of ballooning to one row per response-pair (3315 rows for 1000
# prompts, in an earlier version of this script).
all_prompts = ds_pr["prompt"]
rows_by_prompt = {}
for i, p in enumerate(all_prompts):
    rows_by_prompt.setdefault(p, []).append(i)
unique_prompts = list(rows_by_prompt.keys())

if len(unique_prompts) < N_TRAIN:
    raise ValueError(f"Not enough unique prompts. Found {len(unique_prompts)}, need {N_TRAIN}.")

random.seed(SEED)
train_prompts = set(random.sample(unique_prompts, N_TRAIN))

train_idx = sorted(rows_by_prompt[p][0] for p in train_prompts)
test_idx = sorted(rows_by_prompt[p][0] for p in unique_prompts if p not in train_prompts)

ds_train = ds_pr.select(train_idx)
ds_test = ds_pr.select(test_idx)

# Guard against regressions: train and test must share no prompts.
train_prompt_set = set(ds_train["prompt"])
test_prompt_set = set(ds_test["prompt"])
overlap = train_prompt_set & test_prompt_set
assert not overlap, f"Data leakage: {len(overlap)} prompts appear in both train and test."

print(f"Train rows: {len(ds_train)} ({len(train_prompt_set)} unique prompts)")
print(f"Test rows:  {len(ds_test)} ({len(test_prompt_set)} unique prompts)")

# Save HF datasets to disk.
out_dir = "pku_saferlhf30k_safety_prompt_response_split"
os.makedirs(out_dir, exist_ok=True)

train_dir = os.path.join(out_dir, "train")
test_dir = os.path.join(out_dir, "test")

ds_train.save_to_disk(train_dir)
ds_test.save_to_disk(test_dir)
print(f"Saved HF train: {train_dir}")
print(f"Saved HF test:  {test_dir}")


# Emit JSON and CSV copies for easy inspection / training pipelines.
def save_json_csv(ds_split, prefix: str):
    rows = [{"prompt": r["prompt"], "response": r["response"]} for r in ds_split]

    json_path = f"{prefix}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"Saved JSON: {json_path}")

    csv_path = f"{prefix}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["prompt", "response"])
        writer.writerows([[clean_ws(r["prompt"]), clean_ws(r["response"])] for r in rows])
    print(f"Saved CSV: {csv_path}")


save_json_csv(ds_train, "pku_saferlhf30k_safety_train_prompt_response")
save_json_csv(ds_test, "pku_saferlhf30k_safety_test_prompt_response")
