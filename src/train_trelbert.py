from pathlib import Path
import json
import numpy as np
import torch

from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from config import ROOT, SEED, TRELBERT_MODEL_ID

DATASET_ID = "ptaszynski/PolishCyberbullyingDataset"
OUT_DIR = ROOT / "models" / "trelbert-cyberbullying-binary"

TEXT_COL = "TEXT"
LABEL_COL = "GENERAL TAG"

MAX_LENGTH = 128


def preprocess_text(text: str) -> str:
    text = str(text)
    text = text.replace("@anonymized_user", "@anonymized_account")
    return text


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", pos_label=1, zero_division=0
    )

    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "harmful_precision": precision,
        "harmful_recall": recall,
        "harmful_f1": f1,
    }


def main():
    set_seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    ds = load_dataset(DATASET_ID)

    train = ds["train"]
    test = ds["test"]

    train = train.map(lambda x: {"text": preprocess_text(x[TEXT_COL]), "labels": int(x[LABEL_COL])})
    test = test.map(lambda x: {"text": preprocess_text(x[TEXT_COL]), "labels": int(x[LABEL_COL])})

    # validation split only from train; official test remains final test
    split = train.train_test_split(test_size=0.1, seed=SEED, stratify_by_column="labels")
    train_ds = split["train"]
    val_ds = split["test"]
    test_ds = test

    tokenizer = AutoTokenizer.from_pretrained(TRELBERT_MODEL_ID)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
        )

    train_ds = train_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)
    test_ds = test_ds.map(tokenize, batched=True)

    keep_cols = ["input_ids", "attention_mask", "labels"]
    train_ds.set_format("torch", columns=keep_cols)
    val_ds.set_format("torch", columns=keep_cols)
    test_ds.set_format("torch", columns=keep_cols)

    model = AutoModelForSequenceClassification.from_pretrained(
        TRELBERT_MODEL_ID,
        num_labels=2,
        id2label={0: "non-harmful", 1: "harmful"},
        label2id={"non-harmful": 0, "harmful": 1},
    )

    args = TrainingArguments(
        output_dir=str(OUT_DIR),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=4,
        weight_decay=0.01,
        warmup_ratio=0.1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        seed=SEED,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()

    val_metrics = trainer.evaluate(val_ds)
    test_metrics = trainer.evaluate(test_ds)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(OUT_DIR))
    tokenizer.save_pretrained(str(OUT_DIR))

    run_log = {
        "base_model": TRELBERT_MODEL_ID,
        "dataset": DATASET_ID,
        "text_col": TEXT_COL,
        "label_col": LABEL_COL,
        "max_length": MAX_LENGTH,
        "seed": SEED,
        "learning_rate": 2e-5,
        "batch_size_train": 16,
        "batch_size_eval": 32,
        "epochs": 4,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "checkpoint_path": str(OUT_DIR),
    }

    with open(OUT_DIR / "training_run.json", "w", encoding="utf-8") as f:
        json.dump(run_log, f, indent=2, ensure_ascii=False)

    print("Saved model to:", OUT_DIR)
    print("Validation:", val_metrics)
    print("Test:", test_metrics)


if __name__ == "__main__":
    main()