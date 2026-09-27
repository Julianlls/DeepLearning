"""Static texts and HTML fragments of the page."""
from app.settings import DATASET_URL, GITHUB_URL, MODEL_FT
from src.config import MODEL_BASE

HERO = f"""
<div class="hero">
  <div class="eyebrow">DSTI · Deep Learning project</div>
  <h1>How much does quantization cost a language model?</h1>
  <p>We compress <strong>flan-t5</strong> from 32-bit floats down to 4-bit integers and measure
  how well it still answers questions on <a href="{DATASET_URL}" target="_blank">AdversarialQA</a>,
  a dataset written by humans trying to fool AI models.</p>
  <div class="kpis">
    <div class="kpi"><div class="kpi-value">±0.001</div><div class="kpi-label">F1 change with BF16 &amp; INT8: halving or quartering memory is free</div></div>
    <div class="kpi"><div class="kpi-value">−0.024</div><div class="kpi-label">F1 lost at INT4 (base), the first real cost</div></div>
    <div class="kpi"><div class="kpi-value">+0.017</div><div class="kpi-label">F1 gained by fine-tuning on the task</div></div>
    <div class="kpi"><div class="kpi-value">12</div><div class="kpi-label">configurations evaluated on 3,000 test questions</div></div>
  </div>
</div>
"""

PLACEHOLDER = '<div class="placeholder">Answers will appear here, one card per precision.</div>'

HOW_IT_WORKS = """
1. Paste a text in **Context** and ask a **Question** whose answer is in the text, or pick an example below.
2. Choose the models and precisions to compare. By default, the raw and fine-tuned models run side by side.
3. Optionally fill **Expected answer** to see which configuration gets it right (scored with the project's F1 metric).

**Precisions:** FP32 is the full model (32-bit floats). BF16 halves the memory (16-bit floats).
INT8 stores the weights as 8-bit integers, about 3× smaller here.
"""

BENCHMARK_INTRO = (
    "Full evaluation on the 3,000-question test split (Colab T4 / A100, "
    "INT8 and INT4 through bitsandbytes)."
)

PREDICTIONS_INTRO = (
    "Answers of the raw flan-t5-base on the first 300 test questions, FP32 vs BF16. "
    "**Click a row to load it in the live demo.**"
)

ABOUT = f"""
### The project
We measure the accuracy cost of quantization (FP32 → BF16 → INT8 → INT4) for
**flan-t5-base** (250M parameters) and **flan-t5-large** (780M) on
[AdversarialQA]({DATASET_URL}). flan-t5-base was also fine-tuned on the task
([checkpoint](https://huggingface.co/{MODEL_FT})).

### Key findings
* **BF16 and INT8 are free**: within ±0.001 F1 of FP32 for every model.
* **INT4 is the first real cost**: −0.024 F1 (base), −0.016 (large), −0.012 (fine-tuned).
  The larger and the fine-tuned models are more robust to 4-bit quantization.
* **Fine-tuning helps modestly**: +0.017 F1 on the held-out test split.

### About this demo
* It runs on CPU in a Docker container. The benchmark used **bitsandbytes** INT8/INT4, which
  require a CUDA GPU, so the live INT8 option uses **PyTorch dynamic quantization** of the
  linear layers (INT8 weights, activations quantized on the fly). INT4 figures come from the benchmark.
* The prompt, the input length (512 tokens) and the greedy decoding (at most 48 new tokens) are
  the same as in the evaluation notebooks: the app imports them from the project's `src/` package.
* Metrics: SQuAD-style Exact Match and token-level F1.
"""

FOOTER = f"""
<div class="footer">
  <a href="{GITHUB_URL}" target="_blank">Source code</a> ·
  <a href="https://huggingface.co/{MODEL_BASE}" target="_blank">flan-t5-base</a> ·
  <a href="https://huggingface.co/{MODEL_FT}" target="_blank">Fine-tuned checkpoint</a> ·
  <a href="{DATASET_URL}" target="_blank">AdversarialQA</a>
</div>
"""
