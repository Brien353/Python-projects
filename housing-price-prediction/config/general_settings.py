from pathlib import Path
# ==========================================
# 1. PATH CONFIGURATION
# ==========================================
 
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_FOLDER = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_FOLDER = PROJECT_ROOT / "data" / "processed"
RAW_DATA_PATH = RAW_DATA_FOLDER / "Housing.csv"
SEGMENTED_DATA_PATH = PROCESSED_DATA_FOLDER / "Housing_segmented.csv"


# ==========================================
# 2. FEATURE SCHEMA
# ==========================================
NUM_CL_FEAT = [
    'area',
    'bedrooms',
    'bathrooms',
    'stories',
    'parking'
]

CAT_CL_FEAT = [
    'mainroad',
    'guestroom',
    'basement',
    'hotwaterheating',
    'airconditioning',
    'prefarea',
    'furnishingstatus'
]