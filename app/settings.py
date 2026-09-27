"""Demo settings: models, precisions, data files and links."""
from pathlib import Path

from src.config import MODEL_BASE

TITLE = "Project Deep Learning DSTI"

APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
STYLE_FILE = APP_DIR / "style.css"

MODEL_FT = "Smambu/flan-t5-base-adversarialqa-ft"
MODELS = {
    "Raw": MODEL_BASE,
    "Fine-tuned": MODEL_FT,
}

PRECISIONS = ["FP32", "BF16", "INT8"]
PRECISION_BITS = {"FP32": "32-bit float", "BF16": "16-bit float", "INT8": "8-bit integer"}

# Test questions where the raw and fine-tuned models disagree
DEMO_EXAMPLE_IDS = [282, 189, 196, 292, 231, 5, 83]

GITHUB_URL = "https://github.com/Julianlls/DeepLearning"
DATASET_URL = "https://huggingface.co/datasets/UCLNLP/adversarial_qa"

# Chart colors, readable on light and dark backgrounds
SERIES_COLORS = ["#6366f1", "#14b8a6", "#f59e0b"]
AXIS_COLOR = "#8b8fa3"
