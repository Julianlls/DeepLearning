from datasets import DatasetDict, load_dataset, load_from_disk
from transformers import AutoTokenizer

from src.config import (
    DATASET_NAME, DATASET_CONFIG, DATA_RAW, DATA_PROCESSED,
    MODEL_BASE, PROMPT_TEMPLATE,
    MAX_INPUT_LENGTH, MAX_TARGET_LENGTH,
    SEED, VAL_SIZE,
)


def load_splits():
    "Download the dataset and build the train/val/test splits."
    raw = load_dataset(DATASET_NAME, DATASET_CONFIG, cache_dir=DATA_RAW)
    split = raw["train"].train_test_split(test_size=VAL_SIZE, seed=SEED)
    return DatasetDict({
        "train": split["train"],
        "val": split["test"],
        "test": raw["validation"],
    })


def format_example(example):
    "Build the prompt string and the target string for one example."
    return {
        "input_text": PROMPT_TEMPLATE.format(
            question=example["question"],
            context=example["context"],
        ),
        "target_text": example["answers"]["text"][0],
    }


def tokenize_example(batch, tokenizer):
    "Tokenize a batch of prompts and targets."
    model_inputs = tokenizer(
        batch["input_text"],
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
    )
    labels = tokenizer(
        text_target=batch["target_text"],
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def build_processed_dataset():
    "Run the full pipeline and save the result to data/processed/."
    tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE)

    ds = load_splits()
    ds = ds.map(format_example)
    ds = ds.map(tokenize_example, batched=True, fn_kwargs={"tokenizer": tokenizer})

    ds.save_to_disk(DATA_PROCESSED)
    return ds


def load_processed():
    "Load the processed dataset, building it first if it doesn't exist yet."
    if DATA_PROCESSED.exists():
        return load_from_disk(DATA_PROCESSED)
    return build_processed_dataset()