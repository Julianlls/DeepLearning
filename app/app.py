"""Gradio demo: effect of quantization on flan-t5 for question answering (AdversarialQA).

Runs on CPU (Docker container, see Dockerfile). bitsandbytes INT8/INT4 need CUDA, so the
live demo uses PyTorch dynamic INT8 quantization instead; the bitsandbytes grid from
the notebooks is shown in the "Benchmark results" tab.

Run locally from the repo root:  python -m app.app
"""
import io
import json
import os
import threading
import time
import warnings
from pathlib import Path

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.config import MAX_GEN_TOKENS, MAX_INPUT_LENGTH, MODEL_BASE, PROMPT_TEMPLATE
from src.evaluate import f1_score

warnings.filterwarnings("ignore")

APP_DIR = Path(__file__).parent
MODEL_FT = "Smambu/flan-t5-base-adversarialqa-ft"
MODELS = {
    "flan-t5-base (raw)": MODEL_BASE,
    "flan-t5-base (fine-tuned)": MODEL_FT,
}
PRECISIONS = ["FP32", "BF16", "INT8 (CPU dynamic)"]

RESULTS = pd.read_csv(APP_DIR / "data" / "results.csv")
EXAMPLES = json.loads((APP_DIR / "data" / "examples.json").read_text(encoding="utf-8"))

# Both models share the flan-t5 tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE)

_cache = {}
_lock = threading.Lock()


def _size_mb(model) -> float:
    # Serialized size, so packed INT8 weights are counted too
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return len(buffer.getvalue()) / 1e6


def load_model(model_label: str, precision: str):
    key = (model_label, precision)
    with _lock:
        if key not in _cache:
            repo = MODELS[model_label]
            if precision == "BF16":
                model = AutoModelForSeq2SeqLM.from_pretrained(repo, dtype=torch.bfloat16)
            else:
                model = AutoModelForSeq2SeqLM.from_pretrained(repo)
                if precision.startswith("INT8"):
                    model = torch.ao.quantization.quantize_dynamic(
                        model, {torch.nn.Linear}, dtype=torch.qint8
                    )
            model.eval()
            _cache[key] = (model, _size_mb(model))
        return _cache[key]


def answer(context: str, question: str, model_label: str, precisions: list):
    if not context.strip() or not question.strip():
        raise gr.Error("Please provide both a context and a question.")
    if not precisions:
        raise gr.Error("Select at least one precision.")

    prompt = PROMPT_TEMPLATE.format(question=question.strip(), context=context.strip())
    inputs = tokenizer(prompt, max_length=MAX_INPUT_LENGTH, truncation=True, return_tensors="pt")

    rows = []
    for precision in PRECISIONS:
        if precision not in precisions:
            continue
        model, size_mb = load_model(model_label, precision)
        start = time.perf_counter()
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, max_new_tokens=MAX_GEN_TOKENS, do_sample=False, num_beams=1
            )
        latency_ms = (time.perf_counter() - start) * 1000
        rows.append({
            "precision": precision,
            "answer": tokenizer.decode(output_ids[0], skip_special_tokens=True),
            "latency (ms)": round(latency_ms),
            "model size (MB)": round(size_mb),
        })
    return pd.DataFrame(rows)


def results_figure(metric: str):
    labels = {"f1": "F1", "exact_match": "Exact Match"}
    precisions = ["FP32", "BF16", "INT8", "INT4"]
    models = RESULTS["model"].unique()
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(8, 4))
    for i, model in enumerate(models):
        scores = RESULTS[RESULTS["model"] == model].set_index("precision").loc[precisions, metric]
        xs = [p + (i - (len(models) - 1) / 2) * width for p in range(len(precisions))]
        bars = ax.bar(xs, scores, width, label=model)
        ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
    ax.set_xticks(range(len(precisions)), precisions)
    ax.set_ylabel(labels[metric])
    ax.set_ylim(0.3, 0.75)
    ax.set_title(f"{labels[metric]} on AdversarialQA test (3,000 examples)")
    ax.legend(loc="upper right", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def results_table():
    df = RESULTS.copy()
    fp32 = df[df["precision"] == "FP32"].set_index("model")["f1"]
    df["F1 vs FP32"] = (df["f1"] - df["model"].map(fp32)).round(3)
    return df


def examples_table(only_disagreements: bool):
    df = pd.DataFrame(EXAMPLES)[["id", "question", "gold", "pred_fp32", "pred_bf16", "f1_fp32", "f1_bf16"]]
    if only_disagreements:
        df = df[df["pred_fp32"] != df["pred_bf16"]]
    return df


def pick_example(evt: gr.SelectData, table: pd.DataFrame):
    example = EXAMPLES[int(table.iloc[evt.index[0]]["id"])]
    return example["context"], example["question"], gr.Tabs(selected="demo")


INTRO = """
# Quantization of flan-t5 on AdversarialQA
How much answer quality do we lose when a model is stored with fewer bits?
Ask a question about a text and compare the answers, latency and memory footprint of the
same model at several numerical precisions.
"""

ABOUT = """
### Project
We measure the accuracy cost of quantization (FP32 → BF16 → INT8 → INT4) for
**flan-t5-base** (250M parameters) and **flan-t5-large** (780M) on
[AdversarialQA](https://huggingface.co/datasets/UCLNLP/adversarial_qa), a question answering
dataset written by humans trying to fool models. flan-t5-base was also fine-tuned on the
task ([checkpoint](https://huggingface.co/Smambu/flan-t5-base-adversarialqa-ft)).

### Key findings
* **BF16 and INT8 are free**: within ±0.001 F1 of FP32 for every model.
* **INT4 is the first real cost**: −0.024 F1 (base), −0.016 (large), −0.012 (fine-tuned).
  The larger and the fine-tuned models are more robust to 4-bit quantization.
* **Fine-tuning helps modestly**: +0.017 F1 on the held-out test split.

### About this demo
* Runs on CPU in a Docker container. The benchmark used **bitsandbytes** INT8/INT4, which require a
  CUDA GPU, so the live INT8 option uses **PyTorch dynamic quantization** of the linear
  layers (weights INT8, activations quantized on the fly).
* Prompt, input length (512 tokens) and greedy decoding (max 48 new tokens) are the same as
  in the evaluation notebooks: the app imports them from the project's `src/` package.
* Metrics: SQuAD-style Exact Match and token-level F1.
"""

with gr.Blocks(title="flan-t5 quantization demo") as demo:
    gr.Markdown(INTRO)

    with gr.Tabs() as tabs:
        with gr.Tab("Live demo", id="demo"):
            with gr.Row():
                with gr.Column(scale=3):
                    context_box = gr.Textbox(label="Context", lines=10)
                    question_box = gr.Textbox(label="Question")
                with gr.Column(scale=2):
                    model_radio = gr.Radio(
                        list(MODELS), value="flan-t5-base (fine-tuned)", label="Model"
                    )
                    precision_boxes = gr.CheckboxGroup(PRECISIONS, value=PRECISIONS, label="Precisions")
                    run_button = gr.Button("Answer", variant="primary")
                    gr.Markdown("First run of a configuration downloads and loads the model (~20 s).")
            output_table = gr.Dataframe(label="Answers", interactive=False)
            gr.Examples(
                examples=[[EXAMPLES[i]["context"], EXAMPLES[i]["question"]] for i in (0, 10, 17, 27, 35)],
                inputs=[context_box, question_box],
                label="Examples from the AdversarialQA test split",
            )
            run_button.click(
                answer,
                inputs=[context_box, question_box, model_radio, precision_boxes],
                outputs=output_table,
            )

        with gr.Tab("Benchmark results", id="results"):
            gr.Markdown(
                "Full evaluation on the 3,000-question test split (Colab T4 / A100, "
                "INT8 and INT4 through bitsandbytes)."
            )
            metric_radio = gr.Radio(
                [("F1", "f1"), ("Exact Match", "exact_match")], value="f1", label="Metric"
            )
            plot = gr.Plot(value=results_figure("f1"))
            metric_radio.change(results_figure, inputs=metric_radio, outputs=plot)
            gr.Dataframe(value=results_table(), interactive=False)

        with gr.Tab("Stored predictions", id="predictions"):
            gr.Markdown(
                "Predictions of raw flan-t5-base on the first 300 test questions, FP32 vs BF16. "
                "Click a row to load it in the live demo."
            )
            disagree_box = gr.Checkbox(label="Only show questions where FP32 and BF16 disagree")
            predictions_table = gr.Dataframe(value=examples_table(False), interactive=False, wrap=True)
            disagree_box.change(examples_table, inputs=disagree_box, outputs=predictions_table)
            predictions_table.select(
                pick_example, inputs=predictions_table, outputs=[context_box, question_box, tabs]
            )

        with gr.Tab("About", id="about"):
            gr.Markdown(ABOUT)


def preload_models():
    # Load every configuration once so the first visitor does not wait
    for model_label in MODELS:
        for precision in PRECISIONS:
            load_model(model_label, precision)
    print("All models loaded", flush=True)


if __name__ == "__main__":
    if os.environ.get("PRELOAD_MODELS") == "1":
        threading.Thread(target=preload_models, daemon=True).start()
    demo.launch()
