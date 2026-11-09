"""
Aggregates all per-run run_result.json files (written by
scripts/run_n1_run.py) into:
  - analysis/results_relevant_n1_summary.csv/json
  - analysis/results_unrelated_n1_summary.csv/json
  - analysis/experiment1_paired.csv/json (one row per run_id, both conditions
    side by side -- the input every Experiment 1 stats/plot script reads)

CPU-only, lightweight (reads small JSON files) -- still submitted via
sbatch per the project's compute policy (see jobs/run_analysis.job).
"""
import argparse
import csv
import json
from pathlib import Path

N1_ROOT = Path(__file__).resolve().parents[1]


def load_condition_results(condition_dir, n_runs):
    rows = []
    missing = []
    for run_id in range(n_runs):
        result_path = condition_dir / f"run_{run_id:02d}" / "run_result.json"
        if not result_path.exists():
            missing.append(run_id)
            continue
        with open(result_path) as f:
            data = json.load(f)
        if not data.get("complete"):
            missing.append(run_id)
            continue
        rows.append(data)
    return rows, missing


def write_csv_json(rows, out_stem):
    if not rows:
        print(f"[build_summary_tables] no rows to write for {out_stem}")
        return
    fieldnames = sorted({k for row in rows for k in row.keys() if not isinstance(row[k], dict)})
    with open(out_stem.with_suffix(".csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    with open(out_stem.with_suffix(".json"), "w") as f:
        json.dump(rows, f, indent=2)
    print(f"[build_summary_tables] wrote {len(rows)} rows to {out_stem}.csv/.json")


def build_paired_table(relevant_rows, unrelated_rows, out_stem):
    relevant_by_id = {r["run_id"]: r for r in relevant_rows}
    unrelated_by_id = {r["run_id"]: r for r in unrelated_rows}
    common_ids = sorted(set(relevant_by_id) & set(unrelated_by_id))

    paired_rows = []
    for run_id in common_ids:
        r = relevant_by_id[run_id]
        u = unrelated_by_id[run_id]
        paired_rows.append({
            "run_id": run_id,
            "seed": r["seed"],
            "fairness_relevant": r["fairness_awareness"],
            "fairness_unrelated": u["fairness_awareness"],
            "fairness_diff": r["fairness_awareness"] - u["fairness_awareness"],
            "privacy_relevant": r["privacy_awareness"],
            "privacy_unrelated": u["privacy_awareness"],
            "privacy_diff": r["privacy_awareness"] - u["privacy_awareness"],
            "fairness_delta_vs_origin_relevant": r["fairness_delta_vs_origin"],
            "fairness_delta_vs_origin_unrelated": u["fairness_delta_vs_origin"],
            "privacy_delta_vs_origin_relevant": r["privacy_delta_vs_origin"],
            "privacy_delta_vs_origin_unrelated": u["privacy_delta_vs_origin"],
            "final_coupled_set_size_relevant": r["final_coupled_set_size"],
            "final_coupled_set_size_unrelated": u["final_coupled_set_size"],
        })

    if paired_rows:
        fieldnames = list(paired_rows[0].keys())
        with open(out_stem.with_suffix(".csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in paired_rows:
                writer.writerow(row)
        with open(out_stem.with_suffix(".json"), "w") as f:
            json.dump(paired_rows, f, indent=2)
    print(f"[build_summary_tables] wrote {len(paired_rows)} paired rows to {out_stem}.csv/.json")
    missing_pairs = (set(relevant_by_id) | set(unrelated_by_id)) - set(common_ids)
    if missing_pairs:
        print(f"[build_summary_tables] WARNING: run_ids missing one condition (excluded from pairing): {sorted(missing_pairs)}")
    return paired_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model label from configs/models.yaml.")
    parser.add_argument("--n_runs", type=int, default=25)
    parser.add_argument("--relevant_dir", type=str, default=None, help="Default: results/<model>/relevant_n1")
    parser.add_argument("--unrelated_dir", type=str, default=None, help="Default: results/<model>/unrelated_n1")
    parser.add_argument("--out_dir", type=str, default=None, help="Default: analysis/<model>")
    args = parser.parse_args()

    if args.relevant_dir is None:
        args.relevant_dir = str(N1_ROOT / "results" / args.model / "relevant_n1")
    if args.unrelated_dir is None:
        args.unrelated_dir = str(N1_ROOT / "results" / args.model / "unrelated_n1")
    if args.out_dir is None:
        args.out_dir = str(N1_ROOT / "analysis" / args.model)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    relevant_rows, relevant_missing = load_condition_results(Path(args.relevant_dir), args.n_runs)
    unrelated_rows, unrelated_missing = load_condition_results(Path(args.unrelated_dir), args.n_runs)

    if relevant_missing:
        print(f"[build_summary_tables] relevant: missing/incomplete run_ids: {relevant_missing}")
    if unrelated_missing:
        print(f"[build_summary_tables] unrelated: missing/incomplete run_ids: {unrelated_missing}")

    write_csv_json(relevant_rows, out_dir / "results_relevant_n1_summary")
    write_csv_json(unrelated_rows, out_dir / "results_unrelated_n1_summary")
    build_paired_table(relevant_rows, unrelated_rows, out_dir / "experiment1_paired")


if __name__ == "__main__":
    main()
