from datasets import load_dataset
import csv
import json
import re


# Load PKU-SafeRLHF and keep examples with at least one safe response.
ds = load_dataset("PKU-Alignment/PKU-SafeRLHF-30K", split="train")

ds = ds.filter(lambda x: x["is_response_0_safe"] or x["is_response_1_safe"])


# Choose the safest response available per example.
def pick_response(ex):
    s0 = ex["is_response_0_safe"]
    s1 = ex["is_response_1_safe"]

    if s0 and s1:
        chosen = int(ex["safer_response_id"])   # safer among the two
    elif s0:
        chosen = 0
    else:
        chosen = 1

    return {
        "prompt": ex["prompt"],
        "response": ex[f"response_{chosen}"],
    }

# Reduce to prompt/response only.
ds_pr = ds.map(pick_response, remove_columns=ds.column_names)

# Sample size and RNG seed.
N = 1000
SEED = 42

if len(ds_pr) < N:
    raise ValueError(f"Dataset has only {len(ds_pr)} rows, cannot sample {N}.")

# Deterministic shuffle + subset.
ds_pr = ds_pr.shuffle(seed=SEED).select(range(N))

print(ds_pr[0])
print("Rows (sample):", len(ds_pr))


# Save HF dataset to disk.
out_dir = "pku_saferlhf30k_train_safe_safer_prompt_response_sample1000"
ds_pr.save_to_disk(out_dir)
print("Saved HF dataset to:", out_dir)


# Write CSV for inspection / training pipelines.
rows = list(ds_pr)

csv_path = out_dir + ".csv"
with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f, quoting=csv.QUOTE_ALL)
    writer.writerow(["prompt", "response"])
    writer.writerows([
        [re.sub(r"\s+", " ", r["prompt"]).strip(),
         re.sub(r"\s+", " ", r["response"]).strip()]
        for r in rows
    ])
print("Saved CSV:", csv_path)


# Write JSON copy of rows.
json_path = out_dir + ".json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=False, indent=2)
print("Saved JSON:", json_path)
