#!/usr/bin/env python3
"""
Plot signed divergence heatmap between reproduction and paper baselines.

- Cell color encodes signed delta: Δ = ours - paper
- Diverging colormap centered at 0:
  - negative deltas -> reds (darker = larger magnitude)
  - positive deltas -> greens (darker = larger magnitude)
  - near 0 -> very light (almost neutral)
- Colorbar range matches observed delta range (no min-max normalization).
- Model labels are shown on the top axis.
- SPIN is forced as the last row (bottom).
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle


DEFAULT_METHOD_ORDER: List[str] = ["Origin", "FFT", "Lora", "Dora", "ReFT", "Spin"]
DEFAULT_MODEL_ORDER: List[str] = ["Qwen", "Mistral", "Vicuna", "Llama2", "LLaMA2", "Llama"]
DEFAULT_METRIC_ORDER: List[str] = ["fairness", "privacy"]


def load_results(module_path: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if not module_path.exists():
        raise FileNotFoundError(f"Results file not found: {module_path}")

    spec = importlib.util.spec_from_file_location("results_dict", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]

    expected_a = "REPRODUCTION_BASELINES"
    expected_b = "ORIGINAL_PAPER_BASELINES"
    if not hasattr(module, expected_a) or not hasattr(module, expected_b):
        available = [name for name in dir(module) if name.isupper()]
        raise KeyError(
            f"Expected {expected_a} and {expected_b} in results_dict.py. "
            f"Available top-level names: {available}"
        )

    ours = getattr(module, expected_a)
    paper = getattr(module, expected_b)
    if not isinstance(ours, dict) or not isinstance(paper, dict):
        raise TypeError(f"{expected_a} and {expected_b} must be dicts.")

    return ours, paper


def _ordered_keys(keys: Iterable[str], desired: List[str] | None) -> List[str]:
    keys_list = list(keys)
    if desired:
        ordered = [k for k in desired if k in keys_list]
        remainder = [k for k in keys_list if k not in ordered]
        return ordered + remainder
    return keys_list


def _force_spin_last(methods: List[str]) -> List[str]:
    # Accept either "Spin" or "SPIN"
    spin_keys = [m for m in methods if m.lower() == "spin"]
    if not spin_keys:
        return methods
    spin = spin_keys[-1]
    return [m for m in methods if m != spin] + [spin]


def _extract_value(node: Any) -> float:
    """
    Extract a numeric value from possibly nested dicts.

    Supports:
    - plain numbers
    - dict with "awareness_rate"
    - dict with any numeric leaf (first numeric found)
    """
    if node is None:
        return np.nan
    if isinstance(node, (int, float, np.floating)):
        return float(node)
    if isinstance(node, dict):
        if "awareness_rate" in node:
            return _extract_value(node["awareness_rate"])
        for val in node.values():
            if isinstance(val, (int, float, np.floating)):
                return float(val)
        return np.nan
    return np.nan


def to_frames(
    ours: Dict[str, Any],
    paper: Dict[str, Any],
    allow_partial: bool,
    fmt: str,
    method_order_override: List[str] | None,
    model_order_override: List[str] | None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ours_methods = list(ours.keys())
    paper_methods = list(paper.keys())

    method_intersection = [m for m in ours_methods if m in paper_methods]
    method_only_ours = [m for m in ours_methods if m not in paper_methods]
    method_only_paper = [m for m in paper_methods if m not in ours_methods]

    if (method_only_ours or method_only_paper) and not allow_partial:
        raise ValueError(
            "Method keys do not match. "
            f"Only in ours: {method_only_ours}. Only in paper: {method_only_paper}."
        )

    method_order = method_order_override or DEFAULT_METHOD_ORDER
    methods = _ordered_keys(method_intersection, method_order)
    methods = _force_spin_last(methods)

    # Collect models
    ours_models = set()
    paper_models = set()
    for method in methods:
        ours_models.update((ours.get(method, {}) or {}).keys())
        paper_models.update((paper.get(method, {}) or {}).keys())

    model_intersection = [m for m in ours_models if m in paper_models]
    model_only_ours = [m for m in ours_models if m not in paper_models]
    model_only_paper = [m for m in paper_models if m not in ours_models]

    if (model_only_ours or model_only_paper) and not allow_partial:
        raise ValueError(
            "Model keys do not match. "
            f"Only in ours: {model_only_ours}. Only in paper: {model_only_paper}."
        )

    model_order = model_order_override or DEFAULT_MODEL_ORDER
    models = _ordered_keys(model_intersection, model_order)

    metrics = DEFAULT_METRIC_ORDER

    # Columns: two per model (fairness, privacy)
    columns: List[str] = []
    for model in models:
        for metric in metrics:
            metric_label = "Fairness↑" if metric == "fairness" else "Privacy↑" if metric == "privacy" else metric
            columns.append(f"{model}\n{metric_label}")

    def build_frame(source: Dict[str, Any]) -> pd.DataFrame:
        data: List[List[float]] = []
        for method in methods:
            row: List[float] = []
            method_dict = source.get(method, {}) or {}
            for model in models:
                model_dict = method_dict.get(model, {}) or {}
                for metric in metrics:
                    row.append(_extract_value(model_dict.get(metric)))
            data.append(row)
        return pd.DataFrame(data, index=methods, columns=columns)

    ours_df = build_frame(ours)
    paper_df = build_frame(paper)

    # Validate fmt early
    try:
        _ = format(0.0, fmt)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Invalid number format '{fmt}': {exc}") from exc

    return ours_df, paper_df


def compute_delta(ours: pd.DataFrame, paper: pd.DataFrame) -> Tuple[pd.DataFrame, float, float]:
    delta = ours - paper
    valid = delta.values[~np.isnan(delta.values)]
    if valid.size == 0:
        raise ValueError("No comparable numeric cells between ours and paper.")

    vmin = float(np.min(valid))
    vmax = float(np.max(valid))

    # Ensure TwoSlopeNorm is valid with vcenter=0: need vmin < 0 < vmax.
    if vmin >= 0.0:
        vmin = 0.0
    if vmax <= 0.0:
        vmax = 0.0

    # Edge case: all deltas exactly 0 -> make a tiny range around 0 for plotting
    if vmin == 0.0 and vmax == 0.0:
        vmin, vmax = -1e-12, 1e-12

    return delta, vmin, vmax


def make_signed_diverging_colormap() -> LinearSegmentedColormap:
    # Dark red -> near-white -> dark green
    # Near 0 should look very light/neutral.
    return LinearSegmentedColormap.from_list(
        "signed_delta",
        ["#8e1b10", "#fbfbfb", "#0b6e2f"],
        N=256,
    )


def plot_heatmap(
    ours: pd.DataFrame,
    paper: pd.DataFrame,
    delta: pd.DataFrame,
    vmin: float,
    vmax: float,
    title: str | None,
    outdir: Path,
    fmt: str,
    show_delta_in_cell: bool,
) -> Tuple[Path, Path]:
    sns.set_theme(style="white")

    # Slightly wider for 3+ models * 2 metrics
    fig, ax = plt.subplots(figsize=(11.5, 4.6))

    mask = ours.isna() | paper.isna()
    cmap = make_signed_diverging_colormap()
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

    heat = sns.heatmap(
        delta,
        cmap=cmap,
        norm=norm,
        linewidths=0.6,
        linecolor="#ffffff",
        mask=mask,
        cbar=True,
        ax=ax,
    )

    cbar = heat.collections[0].colorbar
    cbar.set_label("Δ = reproduction − paper")

    # Shade missing values with hatched grey
    nrows, ncols = ours.shape
    for i in range(nrows):
        for j in range(ncols):
            if mask.iat[i, j]:
                rect = Rectangle(
                    (j, i),
                    1,
                    1,
                    facecolor="#eeeeee",
                    edgecolor="#ffffff",
                    hatch="///",
                    linewidth=0.6,
                )
                ax.add_patch(rect)

    # Cell annotations: ours and (paper), optionally delta
    for i in range(nrows):
        for j in range(ncols):
            ours_val = ours.iat[i, j]
            paper_val = paper.iat[i, j]
            delta_val = delta.iat[i, j]

            if np.isnan(ours_val) or np.isnan(paper_val):
                text = "NA"
            else:
                base = f"{format(float(ours_val), fmt)}\n({format(float(paper_val), fmt)})"
                if show_delta_in_cell and not np.isnan(delta_val):
                    base = f"{base}\nΔ {format(float(delta_val), fmt)}"
                text = base

            ax.text(
                j + 0.5,
                i + 0.5,
                text,
                ha="center",
                va="center",
                fontsize=9,
                color="#111111",
            )

    # Put model names at top axis
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_xlabel("")  # paper-like tables usually omit x-axis title
    ax.set_ylabel("")

    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="left")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    if title:
        ax.set_title(title, pad=28)

    fig.tight_layout()

    outdir.mkdir(parents=True, exist_ok=True)
    png_path = outdir / "baselines_delta_heatmap.png"
    pdf_path = outdir / "baselines_delta_heatmap.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)

    logging.info("Saved: %s", png_path)
    logging.info("Saved: %s", pdf_path)

    return png_path, pdf_path


def parse_list_arg(value: str | None) -> List[str] | None:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot signed divergence heatmap for baselines.")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(__file__).parent,
        help="Output directory for figures (default: script directory).",
    )
    parser.add_argument("--title", type=str, default=None, help="Optional figure title.")
    parser.add_argument(
        "--allow_partial",
        action="store_true",
        help="Allow partial overlap between ours and paper keys.",
    )
    parser.add_argument(
        "--fmt",
        type=str,
        default=".3f",
        help="Number format for annotations (default: .3f).",
    )
    parser.add_argument(
        "--show_delta_in_cell",
        action="store_true",
        help="Include signed delta in cell annotations.",
    )
    parser.add_argument(
        "--model_order",
        type=str,
        default=None,
        help="Comma-separated model order override.",
    )
    parser.add_argument(
        "--method_order",
        type=str,
        default=None,
        help="Comma-separated method order override. SPIN will still be forced last.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args()

    results_path = Path(__file__).with_name("results_dict.py")
    ours_dict, paper_dict = load_results(results_path)

    ours_df, paper_df = to_frames(
        ours_dict,
        paper_dict,
        allow_partial=args.allow_partial,
        fmt=args.fmt,
        method_order_override=parse_list_arg(args.method_order),
        model_order_override=parse_list_arg(args.model_order),
    )

    delta_df, vmin, vmax = compute_delta(ours_df, paper_df)

    png_path, pdf_path = plot_heatmap(
        ours_df,
        paper_df,
        delta_df,
        vmin=vmin,
        vmax=vmax,
        title=args.title,
        outdir=args.outdir,
        fmt=args.fmt,
        show_delta_in_cell=args.show_delta_in_cell,
    )

    valid = delta_df.values[~np.isnan(delta_df.values)]
    logging.info("Summary: shape=%s, observed_delta_min=%.6f, observed_delta_max=%.6f", ours_df.shape, float(np.min(valid)), float(np.max(valid)))
    print(f"Wrote: {png_path}")
    print(f"Wrote: {pdf_path}")


if __name__ == "__main__":
    main()