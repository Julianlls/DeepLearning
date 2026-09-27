"""Gradio demo: effect of quantization on flan-t5 for question answering (AdversarialQA).

Layout and event wiring only. The rest of the app lives in:
    settings.py   models, precisions, links
    inference.py  model loading and generation (CPU)
    data.py       benchmark results and stored predictions
    charts.py     benchmark charts
    cards.py      HTML of the live demo answers
    content.py    static texts
    opengraph.py  link previews (Discord, Slack...)
    style.css     page style

Run locally from the repo root:  python -m app.app
"""
import os
import threading

import gradio as gr
import pandas as pd
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse

from app import content
from app.cards import answer_card, model_column, results_page
from app.charts import results_figure
from app.data import EXAMPLES, examples_table, results_table
from app.inference import encode, generate, preload_models
from app.opengraph import OG_IMAGE_PATH, OpenGraphMiddleware
from app.settings import (
    DEMO_EXAMPLE_IDS, FAVICON_FILE, MODELS, OG_IMAGE_FILE, PRECISIONS, STYLE_FILE, TITLE,
)

# Black and white: no colored label chips, inverted primary button in dark mode
THEME = gr.themes.Base(
    primary_hue="neutral",
    neutral_hue="neutral",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    block_label_background_fill="transparent",
    block_label_background_fill_dark="transparent",
    block_label_text_color="*neutral_500",
    block_label_text_color_dark="*neutral_400",
    block_title_text_color="*neutral_500",
    block_title_text_color_dark="*neutral_400",
    button_primary_background_fill="*neutral_900",
    button_primary_background_fill_hover="*neutral_700",
    button_primary_text_color="white",
    button_primary_background_fill_dark="*neutral_50",
    button_primary_background_fill_hover_dark="*neutral_200",
    button_primary_text_color_dark="*neutral_900",
    checkbox_background_color_selected="*neutral_900",
    checkbox_background_color_selected_dark="*neutral_50",
    # Choice groups look like toggle buttons: outlined when off, filled when on
    checkbox_label_background_fill="transparent",
    checkbox_label_background_fill_dark="transparent",
    checkbox_label_background_fill_hover="*neutral_100",
    checkbox_label_background_fill_hover_dark="*neutral_800",
    checkbox_label_border_width="1px",
    checkbox_label_border_color="*neutral_300",
    checkbox_label_border_color_dark="*neutral_600",
    checkbox_label_border_color_selected="*neutral_900",
    checkbox_label_border_color_selected_dark="*neutral_50",
    checkbox_label_background_fill_selected="*neutral_900",
    checkbox_label_background_fill_selected_dark="*neutral_50",
    checkbox_label_text_color_selected="white",
    checkbox_label_text_color_selected_dark="*neutral_900",
)


def keep_one_selected(selected: list, previous: list):
    """Refuse an empty selection: put back the last selected option."""
    if not selected:
        gr.Warning("At least one option must stay selected.")
        return previous, previous
    return selected, selected


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
            models_box = gr.CheckboxGroup(
                list(MODELS), value=list(MODELS), label="Models",
                info="Click to turn a model on or off", elem_classes="toggle-group",
            )
            precisions_box = gr.CheckboxGroup(
                PRECISIONS, value=PRECISIONS, label="Precisions",
                info="Click to turn a precision on or off", elem_classes="toggle-group",
            )
            run_button = gr.Button("Answer", variant="primary", size="lg")
            with gr.Accordion("How it works", open=False):
                gr.Markdown(content.HOW_IT_WORKS)

    for choices_box in (models_box, precisions_box):
        previous = gr.State(choices_box.value)
        choices_box.input(keep_one_selected, inputs=[choices_box, previous], outputs=[choices_box, previous])

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
        [("F1", "f1"), ("Exact Match", "exact_match")], value="f1", label="Metric",
        container=False, elem_classes="toggle-group",
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


with gr.Blocks(title=TITLE) as demo:
    with gr.Row(elem_classes="header-row"):
        gr.HTML(content.HERO)
        theme_button = gr.Button(
            "Light / Dark", variant="secondary", size="sm", scale=0, min_width=110,
            elem_classes="theme-toggle",
        )
    theme_button.click(None, js=content.TOGGLE_THEME_JS)
    demo.load(None, js=content.RESTORE_THEME_JS)

    with gr.Tabs() as tabs:
        with gr.Tab("Live demo", id="demo"):
            demo_inputs = build_demo_tab()
        with gr.Tab("Benchmark results", id="results"):
            build_results_tab()
        with gr.Tab("Stored predictions", id="predictions"):
            build_predictions_tab(demo_inputs, tabs)
    gr.HTML(content.FOOTER)


def create_server():
    """Gradio mounted in FastAPI, so the page can serve its own link-preview tags and image."""
    server = FastAPI()

    @server.get(OG_IMAGE_PATH, include_in_schema=False)
    def og_image():
        return FileResponse(OG_IMAGE_FILE, media_type="image/png")

    server = gr.mount_gradio_app(
        server, demo, path="/",
        theme=THEME, css_paths=[STYLE_FILE], favicon_path=FAVICON_FILE, footer_links=[],
    )
    return OpenGraphMiddleware(server)


if __name__ == "__main__":
    if os.environ.get("PRELOAD_MODELS") == "1":
        threading.Thread(target=preload_models, daemon=True).start()
    uvicorn.run(
        create_server(),
        host=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
    )
