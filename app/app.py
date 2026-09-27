"""Gradio demo: effect of quantization on flan-t5 for question answering (AdversarialQA).

Layout and event wiring only. The rest of the app lives in:
    settings.py   models, precisions, links
    inference.py  model loading and generation (CPU)
    data.py       benchmark results and stored predictions
    charts.py     benchmark charts
    cards.py      HTML of the live demo answers
    content.py    static texts
    style.css     page style

Run locally from the repo root:  python -m app.app
"""
import os
import threading

import gradio as gr
import pandas as pd

from app import content
from app.cards import answer_card, model_column, results_page
from app.charts import results_figure
from app.data import EXAMPLES, examples_table, results_table
from app.inference import encode, generate, preload_models
from app.settings import DEMO_EXAMPLE_IDS, MODELS, PRECISIONS, STYLE_FILE

THEME = gr.themes.Soft(
    primary_hue="indigo",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
)


def run_demo(context: str, question: str, gold: str, models: list, precisions: list) -> str:
    if not context.strip() or not question.strip():
        raise gr.Error("Please provide both a context and a question.")
    if not models or not precisions:
        raise gr.Error("Select at least one model and one precision.")

    inputs = encode(context.strip(), question.strip())
    gold = gold.strip()
    columns = [
        model_column(model_label, [
            answer_card(precision, generate(model_label, precision, inputs), gold)
            for precision in PRECISIONS if precision in precisions
        ])
        for model_label in MODELS if model_label in models
    ]
    return results_page(columns, gold)


def pick_example(evt: gr.SelectData, table: pd.DataFrame):
    example = EXAMPLES[int(table.iloc[evt.index[0]]["#"])]
    return example["context"], example["question"], example["gold"], gr.Tabs(selected="demo")


def build_demo_tab():
    with gr.Row(equal_height=False):
        with gr.Column(scale=3):
            context_box = gr.Textbox(label="Context", lines=9, placeholder="Paste a paragraph of text here…")
            question_box = gr.Textbox(label="Question", placeholder="Ask something answered in the text…")
            gold_box = gr.Textbox(
                label="Expected answer (optional)", placeholder="Fill it to score each configuration"
            )
        with gr.Column(scale=2):
            models_box = gr.CheckboxGroup(list(MODELS), value=list(MODELS), label="Models")
            precisions_box = gr.CheckboxGroup(PRECISIONS, value=PRECISIONS, label="Precisions")
            run_button = gr.Button("Answer", variant="primary", size="lg")
            with gr.Accordion("How it works", open=False):
                gr.Markdown(content.HOW_IT_WORKS)

    output_html = gr.HTML(content.PLACEHOLDER)
    gr.Examples(
        examples=[[EXAMPLES[i]["context"], EXAMPLES[i]["question"], EXAMPLES[i]["gold"]] for i in DEMO_EXAMPLE_IDS],
        inputs=[context_box, question_box, gold_box],
        example_labels=[EXAMPLES[i]["question"] for i in DEMO_EXAMPLE_IDS],
        label="Try a test question where the raw and fine-tuned models disagree",
    )

    run_inputs = [context_box, question_box, gold_box, models_box, precisions_box]
    run_button.click(run_demo, inputs=run_inputs, outputs=output_html)
    question_box.submit(run_demo, inputs=run_inputs, outputs=output_html)
    return context_box, question_box, gold_box


def build_results_tab():
    gr.Markdown(content.BENCHMARK_INTRO)
    metric_radio = gr.Radio(
        [("F1", "f1"), ("Exact Match", "exact_match")], value="f1", label="Metric", container=False
    )
    plot = gr.Plot(value=results_figure("f1"), show_label=False)
    metric_radio.change(results_figure, inputs=metric_radio, outputs=plot)
    gr.Dataframe(value=results_table(), interactive=False, show_label=False)


def build_predictions_tab(demo_inputs, tabs):
    gr.Markdown(content.PREDICTIONS_INTRO)
    disagree_box = gr.Checkbox(label="Only show questions where FP32 and BF16 disagree")
    table = gr.Dataframe(
        value=examples_table(), interactive=False, wrap=True, show_label=False,
        column_widths=["5%", "40%", "15%", "15%", "15%", "5%", "5%"],
    )
    disagree_box.change(examples_table, inputs=disagree_box, outputs=table)
    table.select(pick_example, inputs=table, outputs=[*demo_inputs, tabs])


with gr.Blocks(title="flan-t5 quantization demo") as demo:
    gr.HTML(content.HERO)
    with gr.Tabs() as tabs:
        with gr.Tab("🔍 Live demo", id="demo"):
            demo_inputs = build_demo_tab()
        with gr.Tab("📊 Benchmark results", id="results"):
            build_results_tab()
        with gr.Tab("🗂 Stored predictions", id="predictions"):
            build_predictions_tab(demo_inputs, tabs)
        with gr.Tab("ℹ️ About", id="about"):
            gr.Markdown(content.ABOUT)
    gr.HTML(content.FOOTER)


if __name__ == "__main__":
    if os.environ.get("PRELOAD_MODELS") == "1":
        threading.Thread(target=preload_models, daemon=True).start()
    demo.launch(theme=THEME, css_paths=[STYLE_FILE])
