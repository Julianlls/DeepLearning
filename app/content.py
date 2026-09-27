"""Static texts and HTML fragments of the page."""
from app.settings import DATASET_URL, GITHUB_URL, MODEL_FT, TITLE
from src.config import MODEL_BASE

HERO = f"""
<div class="hero">
  <h1>{TITLE}</h1>
  <p>How much does quantization cost a language model? We compress <strong>flan-t5</strong>
  from 32-bit floats down to 4-bit integers and measure how well it still answers questions on
  <a href="{DATASET_URL}" target="_blank">AdversarialQA</a>, a dataset written by humans trying
  to fool AI models.</p>
</div>
"""

PLACEHOLDER = '<div class="placeholder">Answers will appear here, one card per precision.</div>'

HOW_IT_WORKS = """
1. Paste a text in **Context** and ask a **Question** whose answer is in the text, or pick an example below.
2. Choose the models and precisions to compare. By default, the raw and fine-tuned models run side by side.
3. Optionally fill **Expected answer** to see which configuration gets it right (scored with the project's F1 metric).

**Precisions:** FP32 is the full model (32-bit floats). BF16 halves the memory (16-bit floats).
INT8 stores the weights as 8-bit integers, about 3× smaller here. This demo runs on CPU, so INT8
uses PyTorch dynamic quantization; the benchmark used bitsandbytes on GPU.
"""

BENCHMARK_INTRO = (
    "Full evaluation on the 3,000-question test split (Colab T4 / A100, "
    "INT8 and INT4 through bitsandbytes)."
)

PREDICTIONS_INTRO = (
    "Answers of the raw flan-t5-base on the first 300 test questions, FP32 vs BF16. "
    "**Click a row to load it in the live demo.**"
)

FOOTER = f"""
<div class="footer">
  <a href="{GITHUB_URL}" target="_blank">Source code</a> ·
  <a href="https://huggingface.co/{MODEL_BASE}" target="_blank">flan-t5-base</a> ·
  <a href="https://huggingface.co/{MODEL_FT}" target="_blank">Fine-tuned checkpoint</a> ·
  <a href="{DATASET_URL}" target="_blank">AdversarialQA</a>
</div>
"""

# Theme switch: Gradio puts the "dark" class on <body>; the choice is remembered per browser
TOGGLE_THEME_JS = """
() => {
  const dark = document.body.classList.toggle('dark');
  try { localStorage.setItem('theme', dark ? 'dark' : 'light'); } catch (e) {}
}
"""

RESTORE_THEME_JS = """
() => {
  try {
    const theme = localStorage.getItem('theme');
    if (theme) document.body.classList.toggle('dark', theme === 'dark');
  } catch (e) {}
}
"""
