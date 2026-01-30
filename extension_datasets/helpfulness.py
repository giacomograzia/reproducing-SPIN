from datasets import load_dataset
import random
import json
import csv
import re
import os

def clean_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


ds = load_dataset("nvidia/HelpSteer", split="train")
ds_h4 = ds.filter(lambda x: x["helpfulness"] == 4)

N_TRAIN = 1000
seed = 0

if len(ds_h4) < N_TRAIN:
    raise ValueError(f"Not enough rows with helpfulness==4. Found {len(ds_h4)}, need {N_TRAIN}.")

random.seed(seed)
all_idx = list(range(len(ds_h4)))
train_idx = set(random.sample(all_idx, N_TRAIN))
test_idx = [i for i in all_idx if i not in train_idx]
train_idx = sorted(list(train_idx))

ds_train = ds_h4.select(train_idx)
ds_test = ds_h4.select(test_idx)

keep_cols = ["prompt", "response"]
ds_train = ds_train.remove_columns([c for c in ds_train.column_names if c not in keep_cols])
ds_test  = ds_test.remove_columns([c for c in ds_test.column_names  if c not in keep_cols])

print(f"Train rows: {len(ds_train)}")
print(f"Test rows:  {len(ds_test)}")

out_dir = "helpsteer_val_h4_prompt_response_split"
os.makedirs(out_dir, exist_ok=True)

train_dir = os.path.join(out_dir, "train")
test_dir = os.path.join(out_dir, "test")

ds_train.save_to_disk(train_dir)
ds_test.save_to_disk(test_dir)
print(f"Saved HF train: {train_dir}")
print(f"Saved HF test:  {test_dir}")

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

save_json_csv(ds_train, "helpsteer_val_h4_train_prompt_response")
save_json_csv(ds_test,  "helpsteer_val_h4_test_prompt_response")