"""
train.py
Fine-tunes a BERT based Arabic model for PII token classification.

"""
import json
import os
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

import yaml
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
)
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


# CHARACTER OFFSETS : BIO TAGS
def char_spans_to_bio_per_token(
    text: str,
    entities: List[Dict],
    tokenizer,
    max_length: int,
    label2id: Dict[str, int],
) -> Dict:
    """
    Convert (text, char-level entity spans) into tokenized input + per-token BIO labels.

    """
    enc = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    offsets = enc["offset_mapping"]

    # Sort entities by start position; in case of overlap, the earlier one wins.
    sorted_ents = sorted(entities, key=lambda x: (x["start"], -x["end"]))

    labels = []
    for tok_idx, (s, e) in enumerate(offsets):

        if s == 0 and e == 0:
            labels.append(-100)
            continue

        matched_label = None
        is_first_subword_of_entity = False
        for ent in sorted_ents:

            if s < ent["end"] and e > ent["start"]:
                matched_label = ent["label"]

                is_first_subword_of_entity = (s <= ent["start"])
                break

        if matched_label is None:
            labels.append(label2id["O"])
        else:
            prefix = "B-" if is_first_subword_of_entity else "I-"
            labels.append(label2id[f"{prefix}{matched_label}"])

    enc["labels"] = labels
    enc.pop("offset_mapping")  
    return enc


def fix_bio_first_subword(text, entities, tokenizer, max_length, label2id):
    """
    Robust version: explicitly assign B- to the FIRST token of each entity span,
    I- to subsequent tokens, by walking tokens in order and tracking entity state.
    """
    enc = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    offsets = enc["offset_mapping"]
    n_tokens = len(offsets)

    labels = [label2id["O"]] * n_tokens
    for i, (s, e) in enumerate(offsets):
        if s == 0 and e == 0:
            labels[i] = -100

    for ent in entities:
        ent_start, ent_end, ent_label = ent["start"], ent["end"], ent["label"]
        first_tok = None
        last_tok = None
        for i, (s, e) in enumerate(offsets):
            if s == 0 and e == 0:
                continue
            if s < ent_end and e > ent_start:
                if first_tok is None:
                    first_tok = i
                last_tok = i

        if first_tok is None:
            continue 

        labels[first_tok] = label2id[f"B-{ent_label}"]
        for i in range(first_tok + 1, last_tok + 1):
            if labels[i] != -100:
                labels[i] = label2id[f"I-{ent_label}"]

    enc["labels"] = labels
    enc.pop("offset_mapping")
    return enc


def prepare_dataset(examples: List[Dict], tokenizer, max_length, label2id) -> Dataset:
    processed = [
        fix_bio_first_subword(ex["text"], ex["entities"], tokenizer, max_length, label2id)
        for ex in examples
    ]

    keys = processed[0].keys()
    data = {k: [p[k] for p in processed] for k in keys}
    return Dataset.from_dict(data)


# METRICS
def make_compute_metrics(id2label):
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_preds = []
        true_labels = []
        for pred_seq, label_seq in zip(predictions, labels):
            cur_pred = []
            cur_lab = []
            for p_, l_ in zip(pred_seq, label_seq):
                if l_ != -100:
                    cur_pred.append(id2label[p_])
                    cur_lab.append(id2label[l_])
            true_preds.append(cur_pred)
            true_labels.append(cur_lab)

        return {
            "precision": precision_score(true_labels, true_preds, zero_division=0),
            "recall": recall_score(true_labels, true_preds, zero_division=0),
            "f1": f1_score(true_labels, true_preds, zero_division=0),
        }
    return compute_metrics



# MAIN
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/training_config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    label_list = cfg["labels"]
    label2id = {lbl: i for i, lbl in enumerate(label_list)}
    id2label = {i: lbl for lbl, i in label2id.items()}

    base_model = cfg["model"]["base_model"]
    max_length = cfg["model"]["max_length"]

    print(f"Loading tokenizer & model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(
        base_model,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )

    print("Loading data...")
    train_data = load_jsonl(cfg["data"]["train_path"])
    val_data = load_jsonl(cfg["data"]["val_path"])
    print(f"  train: {len(train_data)}, val: {len(val_data)}")

    print("Tokenizing and aligning labels...")
    train_ds = prepare_dataset(train_data, tokenizer, max_length, label2id)
    val_ds = prepare_dataset(val_data, tokenizer, max_length, label2id)

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    t_cfg = cfg["training"]
    training_args = TrainingArguments(
        output_dir=t_cfg["output_dir"],
        num_train_epochs=t_cfg["num_train_epochs"],
        per_device_train_batch_size=t_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=t_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=t_cfg.get("gradient_accumulation_steps", 1),
        learning_rate=t_cfg["learning_rate"],
        weight_decay=t_cfg["weight_decay"],
        warmup_steps=int(t_cfg["warmup_ratio"] * (15000 // t_cfg["per_device_train_batch_size"]) * t_cfg["num_train_epochs"]),
        lr_scheduler_type=t_cfg["lr_scheduler_type"],
        logging_steps=t_cfg["logging_steps"],
        eval_strategy=t_cfg["eval_strategy"],
        save_strategy=t_cfg["save_strategy"],
        save_total_limit=t_cfg["save_total_limit"],
        load_best_model_at_end=t_cfg["load_best_model_at_end"],
        metric_for_best_model=t_cfg["metric_for_best_model"],
        greater_is_better=t_cfg["greater_is_better"],
        seed=t_cfg["seed"],
        fp16=t_cfg.get("fp16", False) and torch.cuda.is_available(),
        label_smoothing_factor=t_cfg.get("label_smoothing_factor", 0.0),
        report_to=t_cfg.get("report_to", "none"),
    )

    callbacks = []
    if t_cfg.get("early_stopping_patience"):
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=t_cfg["early_stopping_patience"]
        ))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=make_compute_metrics(id2label),
        callbacks=callbacks,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving final model to {t_cfg['output_dir']}")
    trainer.save_model(t_cfg["output_dir"])
    tokenizer.save_pretrained(t_cfg["output_dir"])

    # Final eval
    print("\nFinal validation metrics:")
    metrics = trainer.evaluate()
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # Save metrics to file
    with open(Path(t_cfg["output_dir"]) / "val_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
