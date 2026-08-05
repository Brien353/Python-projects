from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yml"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    _raw_config = yaml.safe_load(f)

# Paths
RAW_DATA_FOLDER = PROJECT_ROOT / _raw_config["paths"]["raw_data_dir"]
PROCESSED_DATA_FOLDER = PROJECT_ROOT / _raw_config["paths"]["processed_data_dir"]

RAW_DATA_PATH = PROJECT_ROOT / _raw_config["paths"]["raw_data_file"]
SEGMENTED_DATA_PATH = PROJECT_ROOT / _raw_config["paths"]["segmented_data_file"]

# Features
NUM_CL_FEAT = _raw_config["features"]["numerical"]
CAT_CL_FEAT = _raw_config["features"]["categorical"]