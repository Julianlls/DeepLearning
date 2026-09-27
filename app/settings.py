"""Demo settings: models, precisions, data files and links."""
import os
from pathlib import Path

from src.config import MODEL_BASE

TITLE = "Project Deep Learning DSTI"
DESCRIPTION = (
    "How much does quantization cost a language model? Live demo comparing flan-t5 "
    "in FP32, BF16 and INT8 on the AdversarialQA question answering dataset."
)
# Link previews (Discord, Slack...) need absolute URLs
PUBLIC_URL = os.environ.get("PUBLIC_URL", "https://flant5.sofianechaoui.com").rstrip("/")

APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
STYLE_FILE = APP_DIR / "style.css"
FAVICON_FILE = APP_DIR / "favicon" / "favicon.ico"
OG_IMAGE_FILE = APP_DIR / "static" / "og-image.png"

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
