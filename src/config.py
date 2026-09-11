from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Dataset
DATASET_NAME = "UCLNLP/adversarial_qa"
DATASET_CONFIG = "adversarialQA"

# --- Models (both share the same tokenizer) ---
MODEL_BASE = "google/flan-t5-base"
MODEL_LARGE = "google/flan-t5-large"

# Prompt / length budget (from EDA)
PROMPT_TEMPLATE = "question: {question}  context: {context}"
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 32
MAX_GEN_TOKENS = 48

# Reproducibility
SEED = 42
VAL_SIZE = 3000