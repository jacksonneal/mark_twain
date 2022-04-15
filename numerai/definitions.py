import os

# Project root
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# To store local pytorch lightning logs
LOG_DIR = os.path.join(ROOT_DIR, '../lightning_logs')
# To store local wandb logs
WANDB_LOG_DIR = os.path.join(ROOT_DIR, '../wandb_logs')
# To store numerai data
DATA_DIR = os.path.join(ROOT_DIR, '../datasets')
# Provided feature sets file
PROVIDED_FEATURES_FILE = os.path.join(DATA_DIR, 'features.json')
# Options for provided feature sets
PROVIDED_FEATURE_SETS = ["legacy", "small", "medium"]
# Generated feature sets file
CUSTOM_FEATURES_FILE = os.path.join(DATA_DIR, 'custom_feature_sets.json')
# Run config directory
CONF_DIR = os.path.join(ROOT_DIR, 'config')
# Saved PCA computation directory
PCA_DIR = os.path.join(DATA_DIR, 'pca')

# Static dataset columns
DATA_TYPE_COL = "data_type"
PREDICTION_COL = "prediction"
ERA_COL = "era"
TARGET_COL = "target_nomi_20"
