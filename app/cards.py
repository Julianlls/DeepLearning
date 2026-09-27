"""HTML rendering of the live demo answers: one column per model, one card per precision."""
import html

from app.settings import MODELS, PRECISION_BITS
from src.evaluate import f1_score

VERDICT_LABELS = {"ok": "Correct", "partial": "Partial", "ko": "Wrong"}


def _verdict_badge(answer: str, gold: str) -> str:
    if not gold:
        return ""
    f1 = f1_score(answer, gold)["f1"]
    status = "ok" if f1 == 1 else "partial" if f1 > 0 else "ko"
    return f'<span class="verdict {status}">{VERDICT_LABELS[status]} · F1 {f1:.2f}</span>'


def answer_card(precision: str, result: dict, gold: str) -> str:
    answer = html.escape(result["answer"]) or "<em>(empty answer)</em>"
    return f"""
    <div class="answer-card">
      <div class="card-head">
        <span class="precision">{precision}</span>
        <span class="bits">{PRECISION_BITS[precision]}</span>
        {_verdict_badge(result["answer"], gold)}
      </div>
      <div class="answer-text">{answer}</div>
      <div class="card-stats">
        <span>⏱ {result["latency_ms"]:.0f} ms</span>
        <span>💾 {result["size_mb"]:.0f} MB</span>
      </div>
    </div>"""


def model_column(model_label: str, cards: list) -> str:
    return (
        f'<div class="model-column"><h3>{model_label} model</h3>'
        f'<div class="repo">{MODELS[model_label]}</div>{"".join(cards)}</div>'
    )


def results_page(columns: list, gold: str) -> str:
    expected = (
        f'<div class="expected">Expected answer: <strong>{html.escape(gold)}</strong></div>'
        if gold else ""
    )
    return f'{expected}<div class="results-grid">{"".join(columns)}</div>'
