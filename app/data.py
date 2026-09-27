"""Benchmark results and stored predictions shipped with the app (see app/data/)."""
import json

import pandas as pd

from app.settings import DATA_DIR

RESULTS = pd.read_csv(DATA_DIR / "results.csv")
EXAMPLES = json.loads((DATA_DIR / "examples.json").read_text(encoding="utf-8"))


def results_table() -> pd.DataFrame:
    df = RESULTS.copy()
    fp32 = df[df["precision"] == "FP32"].set_index("model")["f1"]
    df["f1_vs_fp32"] = (df["f1"] - df["model"].map(fp32)).round(3)
    return df.rename(columns={
        "model": "Model", "precision": "Precision", "exact_match": "Exact Match",
        "token_precision": "Token precision", "token_recall": "Token recall",
        "f1": "F1", "f1_vs_fp32": "F1 vs FP32",
    })


def examples_table(only_disagreements: bool = False) -> pd.DataFrame:
    df = pd.DataFrame(EXAMPLES)[["id", "question", "gold", "pred_fp32", "pred_bf16", "f1_fp32", "f1_bf16"]]
    if only_disagreements:
        df = df[df["pred_fp32"] != df["pred_bf16"]]
    return df.rename(columns={
        "id": "#", "question": "Question", "gold": "Expected", "pred_fp32": "FP32 answer",
        "pred_bf16": "BF16 answer", "f1_fp32": "F1 FP32", "f1_bf16": "F1 BF16",
    })
