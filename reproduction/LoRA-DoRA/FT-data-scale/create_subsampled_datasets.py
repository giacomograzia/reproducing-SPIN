#!/usr/bin/env python3
"""
Script to create balanced subsampled datasets for fairness and privacy experiments.

This script reads the full safe datasets (fairness and privacy, 1k samples each) and creates
balanced subsamples for different dataset sizes. For each target size, we draw 
- 50% from fairness and 
- 50% from privacy datasets.

Target sizes (per dataset): 600, 400, 200, 100, 75, 50, 25, 10
Total samples in combined dataset: 1200, 800, 400, 200, 150, 100, 50, 20
"""

import json
import random
from pathlib import Path
from typing import Dict, Any

# Set random seed for reproducibility
RANDOM_SEED = 1234
random.seed(RANDOM_SEED)

# Dataset sizes to create (per dataset - will be 50% from each)
DATASET_SIZES = [600, 400, 200, 100, 75, 50, 25, 10]

# Paths (go up from reproduction/LoRA-DoRA_v2 to project root)
LLAMA_FACTORY_DATA_DIR = Path(__file__).parent.parent.parent / "LlamaFactory" / "data"
PRIVACY_FILE = LLAMA_FACTORY_DATA_DIR / "beaver_train330k_privacy_safe_1k.json"
FAIRNESS_FILE = LLAMA_FACTORY_DATA_DIR / "beaver_train330k_fairness_safe_1k.json"
DATASET_INFO_FILE = LLAMA_FACTORY_DATA_DIR / "dataset_info.json"

def load_dataset(filepath: Path) -> list[Dict[str, Any]]:
    """Load a JSON dataset file."""
    print(f"Loading {filepath.name}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"  Loaded {len(data)} samples")
    return data


def save_dataset(data: list[Dict[str, Any]], filepath: Path) -> None:
    """Save a dataset to JSON file."""
    print(f"Saving {filepath.name} ({len(data)} samples)...")
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  Saved to {filepath}")


def create_balanced_subsample(
    privacy_data: list[Dict[str, Any]],
    fairness_data: list[Dict[str, Any]],
    n_per_dataset: int
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    """
    Create balanced subsample by drawing n_per_dataset samples from each dataset.
    
    Args:
        privacy_data: Full privacy dataset
        fairness_data: Full fairness dataset
        n_per_dataset: Number of samples to draw from each dataset
        
    Returns:
        Tuple of (privacy_subsample, fairness_subsample)
    """
    # Ensure we have enough samples
    if n_per_dataset > len(privacy_data):
        raise ValueError(f"Requested {n_per_dataset} samples but privacy dataset has only {len(privacy_data)}")
    if n_per_dataset > len(fairness_data):
        raise ValueError(f"Requested {n_per_dataset} samples but fairness dataset has only {len(fairness_data)}")
    
    # Random sampling without replacement
    privacy_subsample = random.sample(privacy_data, n_per_dataset)
    fairness_subsample = random.sample(fairness_data, n_per_dataset)
    
    return privacy_subsample, fairness_subsample


def update_dataset_info(dataset_info_path: Path, new_entries: Dict[str, Dict[str, Any]]) -> None:
    """Update dataset_info.json with new dataset entries."""
    print(f"\nUpdating {dataset_info_path.name}...")
    
    # Load existing dataset info
    with open(dataset_info_path, 'r', encoding='utf-8') as f:
        dataset_info = json.load(f)
    
    # Add new entries
    for dataset_name, dataset_config in new_entries.items():
        dataset_info[dataset_name] = dataset_config
        print(f"  Added entry: {dataset_name}")
    
    # Save updated dataset info
    with open(dataset_info_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)
    
    print(f"  Updated {dataset_info_path}")


def main():
    """Main function to create subsampled datasets."""
    print("=" * 80)
    print("Creating balanced subsampled datasets for fairness and privacy experiments")
    print(f"Random seed: {RANDOM_SEED}")
    print("=" * 80)
    print()
    
    # Load original datasets
    privacy_data = load_dataset(PRIVACY_FILE)
    fairness_data = load_dataset(FAIRNESS_FILE)
    
    print()
    print(f"Target dataset sizes (per dataset): {DATASET_SIZES}")
    print(f"Total combined samples: {[size * 2 for size in DATASET_SIZES]}")
    print()
    
    # Track new dataset info entries
    new_dataset_entries = {}
    
    # Create subsampled datasets for each size
    for size in DATASET_SIZES:
        print(f"\n{'=' * 80}")
        print(f"Creating subsamples with {size} samples per dataset ({size * 2} total)")
        print(f"{'=' * 80}")
        
        # Create balanced subsamples
        privacy_subsample, fairness_subsample = create_balanced_subsample(
            privacy_data, fairness_data, size
        )
        
        # Create filenames
        privacy_filename = f"beaver_train330k_privacy_safe_{size}.json"
        fairness_filename = f"beaver_train330k_fairness_safe_{size}.json"
        
        privacy_path = LLAMA_FACTORY_DATA_DIR / privacy_filename
        fairness_path = LLAMA_FACTORY_DATA_DIR / fairness_filename
        
        # Save subsampled datasets
        save_dataset(privacy_subsample, privacy_path)
        save_dataset(fairness_subsample, fairness_path)
        
        # Add entries for dataset_info.json
        privacy_dataset_name = f"beaver_train330k_privacy_safe_{size}"
        fairness_dataset_name = f"beaver_train330k_fairness_safe_{size}"
        
        new_dataset_entries[privacy_dataset_name] = {
            "format": "alpaca",
            "file_name": privacy_filename,
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output"
            }
        }
        
        new_dataset_entries[fairness_dataset_name] = {
            "format": "alpaca",
            "file_name": fairness_filename,
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output"
            }
        }
        
        print(f"✓ Created {privacy_dataset_name} and {fairness_dataset_name}")
    
    # Update dataset_info.json
    update_dataset_info(DATASET_INFO_FILE, new_dataset_entries)
    
    print()
    print("=" * 80)
    print("All subsampled datasets created successfully!")
    print()
    print("Summary:")
    print(f"  - Created {len(DATASET_SIZES)} dataset sizes")
    print(f"  - Total files created: {len(DATASET_SIZES) * 2}")
    print(f"  - Dataset info entries added: {len(new_dataset_entries)}")
    print()
    print("Next steps:")
    print("  1. Review the created datasets in LlamaFactory/data/")
    print("  2. Verify dataset_info.json has been updated")
    print("  3. Run training job loop with the new datasets")
    print("=" * 80)


if __name__ == "__main__":
    main()
