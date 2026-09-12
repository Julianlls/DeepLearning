import json
from collections import Counter
import re
import string
import torch
from tqdm.auto import tqdm

from src.config import EVAL_BATCH_SIZE, MAX_GEN_TOKENS, RESULTS_DIR


def normalize_answer(s: str) -> str:

    def lower(text):
        return text.lower()

    def remove_punc(text):
        exclude = string.punctuation
        return "".join(ch for ch in text if ch not in exclude)

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction: str, target: str) -> int:
    #1 if the two strings match after normalization, 0 otherwise.
    return int(normalize_answer(prediction) == normalize_answer(target))


def f1_score(prediction: str, target: str) -> dict:

    pred_tokens = normalize_answer(prediction).split()
    target_tokens = normalize_answer(target).split()

    # Empty-string guard: full credit only if both sides are empty.
    if len(pred_tokens) == 0 or len(target_tokens) == 0:
        score = float(pred_tokens == target_tokens)
        return {"precision": score, "recall": score, "f1": score}

    common = Counter(pred_tokens) & Counter(target_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    precision = num_same / len(pred_tokens)
    recall = num_same / len(target_tokens)
    f1 = 2 * precision * recall / (precision + recall)

    return {"precision": precision, "recall": recall, "f1": f1}


def compute_metrics(predictions: list, references: list) -> dict:
    
    #Average EM, precision, recall and F1 over a batch
    #predictions: list of strings of all decoded model output per example
    #references:  list of strings of all target answer per example
    
    if len(predictions) != len(references):
        raise ValueError(
            f"Length mismatch: {len(predictions)} predictions "
            f"vs {len(references)} references"
        )

    em_total = 0.0
    precision_total = 0.0
    recall_total = 0.0
    f1_total = 0.0

    for pred, target in zip(predictions, references):
        em_total += exact_match_score(pred, target)
        scores = f1_score(pred, target)
        precision_total += scores["precision"]
        recall_total += scores["recall"]
        f1_total += scores["f1"]

    n = len(predictions)

    return {
        "exact_match": round(em_total / n,2),
        "precision": round(precision_total / n,2),
        "recall": round(recall_total / n,2),
        "f1": round(f1_total / n,2),
        "n_examples": n,
    }


def generate_predictions(model, tokenizer, dataset,
                         batch_size=EVAL_BATCH_SIZE,
                         max_new_tokens=MAX_GEN_TOKENS) -> list:

    # Run greedy generation on a dataset and return the decoded answer strings

    model.eval()
    device = next(model.parameters()).device
    predictions = []

    for start in tqdm(range(0, len(dataset), batch_size), desc="generating"):
        batch = dataset[start:start + batch_size]

        inputs = tokenizer.pad(
            {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
            },
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
            )

        decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        predictions.extend(decoded)

    return predictions


def save_predictions(predictions, model_name, state, precision_mode):
    "Write predictions to results/ under a filename identifying the grid cell"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"preds_{model_name}_{state}_{precision_mode}.json"

    with open(path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    return path


def evaluate_model(model, tokenizer, dataset,
                   model_name, state, precision_mode,
                   batch_size=EVAL_BATCH_SIZE,
                   max_new_tokens=MAX_GEN_TOKENS,
                   save=True):

    predictions = generate_predictions(
        model, tokenizer, dataset,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
    )

    references = [example["text"][0] for example in dataset["answers"]]
    metrics = compute_metrics(predictions, references)

    results_row = {
        "model_name": model_name,
        "state": state,
        "precision_mode": precision_mode,
        **metrics,
    }

    if save:
        save_predictions(predictions, model_name, state, precision_mode)

    return results_row, predictions