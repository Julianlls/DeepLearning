"""Build app/data/examples.json: test-split examples joined with the stored predictions.

The prediction files in results/ are aligned with the test split (original
AdversarialQA validation split), so row i of each file answers test example i.
Rows are fetched from the Hugging Face datasets-server API to avoid downloading
the full dataset.

Usage (from the repo root):  python -m scripts.build_examples
"""
import json

import requests

from src.config import DATASET_CONFIG, DATASET_NAME, PROJECT_ROOT, RESULTS_DIR
from src.evaluate import f1_score

N_EXAMPLES = 300
PAGE = 100
API = "https://datasets-server.huggingface.co/rows"
OUT = PROJECT_ROOT / "app" / "data" / "examples.json"


def fetch_rows(n):
    rows = []
    for offset in range(0, n, PAGE):
        r = requests.get(API, params={
            "dataset": DATASET_NAME, "config": DATASET_CONFIG,
            "split": "validation", "offset": offset, "length": min(PAGE, n - offset),
        }, timeout=60)
        r.raise_for_status()
        rows.extend(item["row"] for item in r.json()["rows"])
    return rows


def main():
    fp32 = json.loads((RESULTS_DIR / "preds_flan-t5-base_raw_fp32.json").read_text(encoding="utf-8"))
    bf16 = json.loads((RESULTS_DIR / "preds_flan-t5-base_raw_bf16.json").read_text(encoding="utf-8"))

    examples = []
    for i, row in enumerate(fetch_rows(N_EXAMPLES)):
        gold = row["answers"]["text"][0]
        examples.append({
            "id": i,
            "question": row["question"],
            "context": row["context"],
            "gold": gold,
            "pred_fp32": fp32[i],
            "pred_bf16": bf16[i],
            "f1_fp32": round(f1_score(fp32[i], gold)["f1"], 3),
            "f1_bf16": round(f1_score(bf16[i], gold)["f1"], 3),
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(examples, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {len(examples)} examples to {OUT}")


if __name__ == "__main__":
    main()
