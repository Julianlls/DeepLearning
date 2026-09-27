"""Model loading at each precision and answer generation, on CPU.

bitsandbytes INT8/INT4 need CUDA, so INT8 here is PyTorch dynamic quantization
of the linear layers.
"""
import io
import threading
import time
import warnings
from functools import cache

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from app.settings import MODELS, PRECISIONS
from src.config import MAX_GEN_TOKENS, MAX_INPUT_LENGTH, MODEL_BASE, PROMPT_TEMPLATE

# torch.ao dynamic quantization and transformers print deprecation notices on every load
warnings.filterwarnings("ignore")

_models = {}
_lock = threading.Lock()


@cache
def get_tokenizer():
    # Both models share the flan-t5 tokenizer
    return AutoTokenizer.from_pretrained(MODEL_BASE)


def _size_mb(model) -> float:
    # Serialized size, so packed INT8 weights are counted too
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return len(buffer.getvalue()) / 1e6


def _build_model(repo: str, precision: str):
    if precision == "BF16":
        return AutoModelForSeq2SeqLM.from_pretrained(repo, dtype=torch.bfloat16)
    model = AutoModelForSeq2SeqLM.from_pretrained(repo)
    if precision == "INT8":
        model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    return model


def load_model(model_label: str, precision: str):
    """Return (model, size in MB), loading it on first use."""
    key = (model_label, precision)
    with _lock:
        if key not in _models:
            model = _build_model(MODELS[model_label], precision).eval()
            _models[key] = (model, _size_mb(model))
        return _models[key]


def preload_models():
    # Load every configuration once so the first visitor does not wait
    for model_label in MODELS:
        for precision in PRECISIONS:
            load_model(model_label, precision)
    print("All models loaded", flush=True)


def encode(context: str, question: str):
    prompt = PROMPT_TEMPLATE.format(question=question, context=context)
    return get_tokenizer()(prompt, max_length=MAX_INPUT_LENGTH, truncation=True, return_tensors="pt")


def generate(model_label: str, precision: str, inputs) -> dict:
    """Greedy decoding, as in the evaluation notebooks."""
    model, size_mb = load_model(model_label, precision)
    start = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=MAX_GEN_TOKENS, do_sample=False, num_beams=1)
    return {
        "answer": get_tokenizer().decode(output_ids[0], skip_special_tokens=True),
        "latency_ms": (time.perf_counter() - start) * 1000,
        "size_mb": size_mb,
    }
