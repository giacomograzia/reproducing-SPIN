"""
Shared model-registry helper: every script in this experiment that needs to
resolve a `--model <label>` CLI argument against configs/models.yaml imports
this instead of duplicating the same load-and-lookup logic. This is the one
place "add a new model" logic lives outside models.yaml itself.
"""
from pathlib import Path

import yaml

N1_ROOT = Path(__file__).resolve().parents[1]  # .../n1_localization_controls
MODELS_YAML_PATH = N1_ROOT / "configs" / "models.yaml"


def load_models(models_yaml_path=MODELS_YAML_PATH):
    with open(models_yaml_path) as f:
        registry = yaml.safe_load(f)
    return registry["models"]


def get_model(label, models_yaml_path=MODELS_YAML_PATH):
    models = load_models(models_yaml_path)
    if label not in models:
        available = ", ".join(sorted(models.keys()))
        raise KeyError(f"Unknown model label '{label}'. Available: {available}. "
                        f"Add a new entry to {models_yaml_path} to register another model.")
    return models[label]
