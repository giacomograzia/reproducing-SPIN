#!/usr/bin/env python3
"""
Extract awareness rates from evaluation results and create comparison plots.

This script:
1. Extracts fairness and privacy awareness rates from eval.out files
2. Creates placeholder dictionaries for FFT and SPIN methods
3. Generates side-by-side plots comparing methods across dataset sizes
"""
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as mpatches
matplotlib.use('Agg')  # Non-interactive backend
from pathlib import Path
from results_dict import SPIN_RESULTS, FFT_RESULTS, LORA_RESULTS, SPIN_EXTRA_RESULTS, SPIN_MULTISEED_RESULTS

OUTPUT_DIR = Path(__file__).parent / "plots"
OUTPUT_DIR.mkdir(exist_ok=True)

# baseline QWEN 7B on safe datasets
def plot_results(spin_results, fft_results, lora_results, origin_fairness=0.715, origin_privacy=0.7671):
    """
    Plot fairness and privacy awareness for each method, using half dataset sizes.
    
    Args:
        spin_results, fft_results, lora_results: Dicts with keys like 'method_size'
        Each value has nested 'fairness' and 'privacy' dicts with 'awareness_rate'
        origin_fairness: Origin model baseline for fairness (default 0.5)
        origin_privacy: Origin model baseline for privacy (default 0.5)
    """
    # Combine all results into one dict
    all_results = {}
    all_results.update(spin_results)
    all_results.update(fft_results)
    all_results.update(lora_results)
    
    # Extract unique dataset sizes and methods
    sizes = set()
    methods = set()
    for key in all_results.keys():
        method, size_str = key.rsplit('_', 1)
        sizes.add(int(size_str))
        methods.add(method)
    
    sizes = sorted(sizes)
    
    # Prepare data for each method and metric - extract awareness_rate from nested dict
    def get_metric(method, metric):
        return [all_results.get(f'{method}_{s}', {}).get(metric, {}).get('awareness_rate', None) for s in sizes]

    # Plot settings - larger figure, thinner lines
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    method_styles = {
        'SPIN': dict(color='#FF8A65', marker='o', linewidth=2.5, markersize=9, label='SPIN'),
        'FFT': dict(color='#BA68C8', marker='X', linewidth=2.5, markersize=9, label='FFT'),
        'LoRA': dict(color='#4FC3F7', marker='s', linewidth=2.5, markersize=7, label='LoRA'),
    }

    # Fairness
    ax = axes[0]
    # Add shaded region for experiments beyond the paper (dataset sizes < 100)
    ax.axvspan(min(sizes), 100, alpha=0.15, color='lightgreen', zorder=0)
    ax.plot(sizes, get_metric('spin', 'fairness'), **method_styles['SPIN'])
    ax.plot(sizes, get_metric('fft', 'fairness'), **method_styles['FFT'])
    ax.plot(sizes, get_metric('lora', 'fairness'), **method_styles['LoRA'])
    ax.axhline(origin_fairness, color='gray', linestyle='--', linewidth=2)
    ax.text(sizes[0] * 0.95, origin_fairness + 0.015, 'Origin Model', color='gray', 
            fontsize=13, va='bottom', ha='left')
    ax.set_title('Awareness of Fairness', fontsize=18, pad=15)
    ax.set_xlabel('Dataset Size', fontsize=15)
    ax.set_ylabel('Awareness Rate', fontsize=15)
    ax.set_xscale('log')
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes], fontsize=12)
    ax.set_ylim(0.4, 0.9)
    ax.tick_params(axis='y', labelsize=12)
    ax.grid(True, alpha=0.3, linewidth=0.8)

    # Privacy
    ax = axes[1]
    # Add shaded region for experiments beyond the paper (dataset sizes < 100)
    ax.axvspan(min(sizes), 100, alpha=0.15, color='lightgreen', zorder=0)
    ax.plot(sizes, get_metric('spin', 'privacy'), **method_styles['SPIN'])
    ax.plot(sizes, get_metric('fft', 'privacy'), **method_styles['FFT'])
    ax.plot(sizes, get_metric('lora', 'privacy'), **method_styles['LoRA'])
    ax.axhline(origin_privacy, color='gray', linestyle='--', linewidth=2)
    ax.text(sizes[0] * 0.95, origin_privacy + 0.015, 'Origin Model', color='gray', 
            fontsize=13, va='bottom', ha='left')
    ax.set_title('Awareness of Privacy', fontsize=18, pad=15)
    ax.set_xlabel('Dataset Size', fontsize=15)
    ax.set_ylabel('Awareness Rate', fontsize=15)
    ax.set_xscale('log')
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes], fontsize=12)
    ax.set_ylim(0.6, 0.9)
    ax.tick_params(axis='y', labelsize=12)
    ax.grid(True, alpha=0.3, linewidth=0.8)

    # Legend (shared)
    handles, labels = axes[1].get_legend_handles_labels()
    # Add custom patch for the green shaded region
    green_patch = mpatches.Patch(color='lightgreen', alpha=0.15, label='Further experiments')
    handles.append(green_patch)
    labels.append('Further experiments')
    fig.legend(handles, labels, alignment= 'center', loc='lower center', ncol=4, fontsize=14, frameon=True, title='Method', title_fontsize=15, bbox_to_anchor=(0.5, -0.06)) # second nr: vertical position
    plt.tight_layout(rect=[0, 0.06, 1, 1])

    # Save
    out_png = OUTPUT_DIR / "awareness_comparison.png"
    out_pdf = OUTPUT_DIR / "awareness_comparison.pdf"
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.close()
    print(f"Saved plots to {out_png} and {out_pdf}")

def plot_spin_extra_results(spin_extra_results):
    """
    Plot fairness and privacy awareness for SPIN extra experiments (smaller sample sizes).
    Both metrics on same plot with different colors.
    
    Args:
        spin_extra_results: Dict with keys like 'spin_variant_size'
        Each value has nested 'fairness' and 'privacy' dicts with 'awareness_rate'
    """
    # Extract unique dataset sizes and variants
    sizes = set()
    variants = set()
    for key in spin_extra_results.keys():
        parts = key.rsplit('_', 1)
        size_str = parts[-1]
        variant = parts[0]
        sizes.add(int(size_str))
        variants.add(variant)
    
    sizes = sorted(sizes)
    variants = sorted(variants)
    
    # Prepare data for each variant and metric
    def get_metric(variant, metric):
        return [spin_extra_results.get(f'{variant}_{s}', {}).get(metric, {}).get('awareness_rate', None) for s in sizes]
    
    # Plot settings - single plot with both fairness and privacy
    fig, ax = plt.subplots(figsize=(6, 4.5))
    
    # Colors for metrics (not orange)
    metric_colors = {
        'fairness': '#E91E63',  # Pink/Magenta
        'privacy': '#009688',   # Teal
    }
    
    # Plot each variant's fairness and privacy
    for variant in variants:
        ax.plot(sizes, get_metric(variant, 'fairness'), 
               color=metric_colors['fairness'],
               marker='o', linewidth=2.5, markersize=9,
               label=f'{variant} Fairness', linestyle='-')
        ax.plot(sizes, get_metric(variant, 'privacy'),
               color=metric_colors['privacy'],
               marker='o', linewidth=2.5, markersize=9,
               label=f'{variant} Privacy', linestyle='--')
    
    ax.set_title('SPIN Extra Experiments - Fairness and Privacy Awareness', fontsize=18, pad=15)
    ax.set_xlabel('Dataset Size', fontsize=15)
    ax.set_ylabel('Awareness Rate', fontsize=15)
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes], fontsize=12)
    ax.set_ylim(0.4, 0.9)
    ax.tick_params(axis='y', labelsize=12)
    ax.grid(True, alpha=0.3, linewidth=0.8)
    ax.legend(fontsize=13, frameon=True, loc='best')
    
    plt.tight_layout()
    
    # Save
    out_png = OUTPUT_DIR / "spin_extra_comparison.png"
    out_pdf = OUTPUT_DIR / "spin_extra_comparison.pdf"
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.close()
    print(f"Saved plots to {out_png} and {out_pdf}")

def plot_spin_focused_comparison(spin_results, spin_extra_results):
    """
    Plot SPIN fairness and privacy from largest sample sizes (1000+) down through extra small samples.
    Uses orange for the main SPIN_RESULTS range, red/teal for smaller extra samples.
    
    Args:
        spin_results: Main SPIN results dict
        spin_extra_results: Extra SPIN results dict with smaller sample sizes
    """
    # Extract sizes from SPIN_RESULTS (main range)
    main_sizes = set()
    for key in spin_results.keys():
        _, size_str = key.rsplit('_', 1)
        main_sizes.add(int(size_str))
    
    main_sizes = sorted(main_sizes)  # Ascending: smallest first
    
    # Extract sizes from SPIN_EXTRA_RESULTS (extra range)
    extra_sizes = set()
    for key in spin_extra_results.keys():
        size_str = key.rsplit('_', 1)[-1]
        extra_sizes.add(int(size_str))
    
    extra_sizes = sorted(extra_sizes)  # Ascending: smallest first
    
    # Prepare main range data
    def get_main_metric(metric):
        return [spin_results.get(f'spin_{s}', {}).get(metric, {}).get('awareness_rate', None) for s in main_sizes]
    
    main_fairness = get_main_metric('fairness')
    main_privacy = get_main_metric('privacy')
    
    # Prepare extra range data
    def get_extra_metric(metric):
        return [spin_extra_results.get(f'spin_{s}', {}).get(metric, {}).get('awareness_rate', None) for s in extra_sizes]
    
    extra_fairness = get_extra_metric('fairness')
    extra_privacy = get_extra_metric('privacy')
    
    # Plot settings
    fig, ax = plt.subplots(figsize=(9, 5.5))
    
    # Get all sizes combined and create evenly spaced x positions
    all_sizes = sorted(set(main_sizes) | set(extra_sizes))  # Ascending
    x_positions = list(range(len(all_sizes)))
    
    # Map size values to x positions
    size_to_x = {size: x for x, size in zip(x_positions, all_sizes)}
    
    # Convert size lists to x positions
    main_x = [size_to_x[s] for s in main_sizes]
    extra_x = [size_to_x[s] for s in extra_sizes]
    
    # Plot main range (1000+) with orange
    ax.plot(main_x, main_fairness,
           color='#FF8A65', marker='o', linewidth=2.5, markersize=9,
           label='Fairness (1000-10)', linestyle='-')
    ax.plot(main_x, main_privacy,
           color='#FF8A65', marker='s', linewidth=2.5, markersize=7,
           label='Privacy (1000-10)', linestyle='--')
    
    # Plot extra range (< 1000) with red/teal
    ax.plot(extra_x, extra_fairness,
           color='#E63946', marker='o', linewidth=2.5, markersize=9,
           label='Fairness (5-1)', linestyle='-')
    ax.plot(extra_x, extra_privacy,
           color='#06A77D', marker='s', linewidth=2.5, markersize=7,
           label='Privacy (5-1)', linestyle='--')
    
    ax.set_title('SPIN - Fairness and Privacy Across All Dataset Sizes', fontsize=18, pad=15)
    ax.set_xlabel('Dataset Size', fontsize=15)
    ax.set_ylabel('Awareness Rate', fontsize=15)
    
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(s) for s in all_sizes], fontsize=12)
    ax.set_ylim(0.4, 0.90)
    ax.tick_params(axis='y', labelsize=12)
    ax.grid(True, alpha=0.3, linewidth=0.8)
    ax.legend(fontsize=12, frameon=True, loc='best')
    
    plt.tight_layout()
    
    # Save
    out_png = OUTPUT_DIR / "spin_focused_comparison.png"
    out_pdf = OUTPUT_DIR / "spin_focused_comparison.pdf"
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.close()
    print(f"Saved focused SPIN plot to {out_png} and {out_pdf}")

def plot_spin_multiseed(multiseed_results, origin_fairness=0.715, origin_privacy=0.7671):
    """
    Plot SPIN fairness and privacy with a shaded ±1 SD band around the mean line.

    Args:
        multiseed_results: Dict[int, list[dict]] — key is dataset size N,
            value is a list of per-seed dicts each with 'fairness' and 'privacy'
            sub-dicts containing 'awareness_rate'.
    """
    import numpy as np

    sizes = sorted(multiseed_results.keys())

    def get_mean_std(metric):
        means, stds = [], []
        for s in sizes:
            rates = [run[metric]['awareness_rate'] for run in multiseed_results[s]]
            means.append(np.mean(rates))
            stds.append(np.std(rates))
        return np.array(means), np.array(stds)

    f_mean, f_std = get_mean_std('fairness')
    p_mean, p_std = get_mean_std('privacy')

    color = '#FF8A65'
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    for ax, means, stds, title, ylim, baseline in [
        (axes[0], f_mean, f_std, 'Awareness of Fairness (SPIN, multi-seed)', (0.4, 0.9), origin_fairness),
        (axes[1], p_mean, p_std, 'Awareness of Privacy (SPIN, multi-seed)',  (0.6, 0.9), origin_privacy),
    ]:
        ax.fill_between(sizes, means - stds, means + stds, color=color, alpha=0.25, label='±1 SD')
        ax.plot(sizes, means, color=color, marker='o', linewidth=2.5, markersize=9, label='SPIN (mean)')
        ax.axhline(baseline, color='gray', linestyle='--', linewidth=2)
        ax.text(sizes[0] * 0.95, baseline + 0.015, 'Origin Model', color='gray',
                fontsize=13, va='bottom', ha='left')
        ax.set_title(title, fontsize=16, pad=15)
        ax.set_xlabel('Dataset Size', fontsize=15)
        ax.set_ylabel('Awareness Rate', fontsize=15)
        ax.set_xscale('log')
        ax.set_xticks(sizes)
        ax.set_xticklabels([str(s) for s in sizes], fontsize=12)
        ax.set_ylim(*ylim)
        ax.tick_params(axis='y', labelsize=12)
        ax.grid(True, alpha=0.3, linewidth=0.8)

    # Deduplicate legend entries (mean line + band)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2,
               fontsize=14, frameon=True, bbox_to_anchor=(0.5, -0.06))
    plt.tight_layout(rect=[0, 0.06, 1, 1])

    out_png = OUTPUT_DIR / "spin_multiseed_shaded.png"
    out_pdf = OUTPUT_DIR / "spin_multiseed_shaded.pdf"
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.close()
    print(f"Saved multi-seed shaded plots to {out_png} and {out_pdf}")


def main():
    print("Plotting awareness results from static dictionaries...")
    plot_results(SPIN_RESULTS, FFT_RESULTS, LORA_RESULTS)
    plot_spin_extra_results(SPIN_EXTRA_RESULTS)
    plot_spin_focused_comparison(SPIN_RESULTS, SPIN_EXTRA_RESULTS)
    if SPIN_MULTISEED_RESULTS:
        plot_spin_multiseed(SPIN_MULTISEED_RESULTS)
    else:
        print("SPIN_MULTISEED_RESULTS is empty — skipping error-bar plot.")

if __name__ == "__main__":
    main()
