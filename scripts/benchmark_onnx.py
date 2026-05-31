"""
benchmark_onnx.py

Benchmarks the ONNX INT8 quantized model on CPU.

"""
import argparse
import json
import time
import platform
import statistics
import re
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

import numpy as np
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForTokenClassification


# SHARED HELPERS 

ARABIC_INDIC = "٠١٢٣٤٥٦٧٨٩"
WESTERN = "0123456789"
NORMALIZE_DIGITS_MAP = str.maketrans(ARABIC_INDIC, WESTERN)
LABEL_TO_TAG = {
    "PERSON": "[PERSON]", "EMAIL": "[EMAIL]", "PHONE_NUMBER": "[PHONE_NUMBER]",
    "ADDRESS": "[ADDRESS]", "ACCOUNT_NUMBER": "[ACCOUNT_NUMBER]",
    "BANK_ACCOUNT_NUMBER": "[BANK_ACCOUNT_NUMBER]", "IBAN": "[IBAN]",
}
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
IBAN_RE  = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")


def _iban_mod97_check(iban: str) -> bool:
    iban = iban.replace(" ", "").upper()
    if len(iban) < 15:
        return False
    rearranged = iban[4:] + iban[:4]
    digits = "".join(str(ord(c) - 55) if c.isalpha() else c for c in rearranged if c.isalnum())
    rem = 0
    for d in digits:
        rem = (rem * 10 + int(d)) % 97
    return rem == 1


def validate_entity(text: str, label: str) -> float:
    if label == "EMAIL":
        return 1.05 if EMAIL_RE.fullmatch(text.strip()) else 0.85
    if label == "IBAN":
        return 1.05 if _iban_mod97_check(text.strip()) else 0.80
    return 1.0


def bio_decode_to_spans(tokens_info: List[Tuple]) -> List[Dict]:
    entities, current = [], None
    for (s, e, label, prob) in tokens_info:
        if label == "O" or label.startswith("-"):
            if current:
                entities.append(current)
                current = None
            continue
        prefix, etype = label[0], label[2:]
        if prefix == "B" or current is None or current["label"] != etype:
            if current:
                entities.append(current)
            current = {"label": etype, "start": s, "end": e, "probs": [prob]}
        else:
            current["end"] = e
            current["probs"].append(prob)
    if current:
        entities.append(current)
    for ent in entities:
        ent["confidence"] = float(np.mean(ent["probs"]))
        del ent["probs"]
    return entities


# ONNX DETECTOR
class ONNXPIIDetector:
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.model = ORTModelForTokenClassification.from_pretrained(model_path)
        self.id2label = self.model.config.id2label
        self.device = "cpu"

    def predict(self, text: str, max_length: int = 256) -> Dict:
        normalized = text.translate(NORMALIZE_DIGITS_MAP)

        enc = self.tokenizer(
            normalized,
            return_offsets_mapping=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        offsets = enc.pop("offset_mapping")[0].tolist()

        outputs = self.model(**enc)
        logits = outputs.logits[0].detach().numpy()
        probs = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
        pred_ids = probs.argmax(axis=-1)
        pred_labels = [self.id2label[int(p)] for p in pred_ids]
        max_probs = probs.max(axis=-1)

        tokens_info = [
            (s, e, pred_labels[i], float(max_probs[i]))
            for i, (s, e) in enumerate(offsets)
            if not (s == 0 and e == 0)
        ]

        raw_entities = bio_decode_to_spans(tokens_info)

        entities = []
        for ent in raw_entities:
            start, end = ent["start"], ent["end"]
            ent_text = text[start:end].rstrip(" .،,;:")
            end -= (len(text[start:end]) - len(ent_text))
            if not ent_text:
                continue
            conf = min(0.999, max(0.0, ent["confidence"] * validate_entity(ent_text, ent["label"])))
            entities.append({
                "text": ent_text, "label": ent["label"],
                "start": start, "end": end, "confidence": round(conf, 4),
            })

        redacted = text
        for ent in sorted(entities, key=lambda x: -x["start"]):
            tag = LABEL_TO_TAG.get(ent["label"], f"[{ent['label']}]")
            redacted = redacted[:ent["start"]] + tag + redacted[ent["end"]:]

        return {"redacted_text": redacted, "entities": entities}


# METRICS
def entity_tuple(e: Dict) -> Tuple:
    return (e["start"], e["end"], e["label"])


def compute_metrics(gold_examples: List[Dict], pred_examples: List[Dict]) -> Dict:
    overall_tp = overall_fp = overall_fn = 0
    per_class = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for gold_ex, pred_ex in zip(gold_examples, pred_examples):
        gold_set = {entity_tuple(e) for e in gold_ex["entities"]}
        pred_set = {entity_tuple(e) for e in pred_ex["entities"]}
        for (_, _, lbl) in gold_set & pred_set:
            overall_tp += 1
            per_class[lbl]["tp"] += 1
        for fp in pred_set - gold_set:
            overall_fp += 1
            per_class[fp[2]]["fp"] += 1
        for fn in gold_set - pred_set:
            overall_fn += 1
            per_class[fn[2]]["fn"] += 1

    def prf(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)}

    overall = prf(overall_tp, overall_fp, overall_fn)
    overall.update({"tp": overall_tp, "fp": overall_fp, "fn": overall_fn})
    per_class_metrics = {
        lbl: {**prf(c["tp"], c["fp"], c["fn"]), "support": c["tp"] + c["fn"]}
        for lbl, c in per_class.items()
    }
    return {"overall": overall, "per_class": per_class_metrics}


# LATENCY

def measure_latency(detector: ONNXPIIDetector, texts: List[str], warmup: int = 10) -> Dict:
    for t in texts[:warmup]:
        detector.predict(t)

    all_times = []
    times_by_length = defaultdict(list)

    for text in texts:
        t0 = time.perf_counter()
        detector.predict(text)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        all_times.append(elapsed_ms)
        n_words = len(text.split())
        bucket = "short (<=25 words)" if n_words <= 25 else "medium (26-60 words)" if n_words <= 60 else "long (>60 words)"
        times_by_length[bucket].append(elapsed_ms)

    def summarize(arr):
        arr = sorted(arr)
        n = len(arr)
        return {
            "count": n,
            "mean_ms": round(statistics.mean(arr), 2),
            "p50_ms":  round(arr[int(n * 0.50)], 2),
            "p95_ms":  round(arr[min(n - 1, int(n * 0.95))], 2),
            "max_ms":  round(max(arr), 2),
        }

    return {
        "hardware": {
            "device": "cpu",
            "platform": platform.platform(),
            "processor": platform.processor() or "unknown",
        },
        "model": "ONNX INT8 quantized",
        "batch_size": 1,
        "overall": summarize(all_times),
        "by_length": {k: summarize(v) for k, v in times_by_length.items()},
        "target_ms": 150,
        "within_target": summarize(all_times)["p95_ms"] < 150,
    }


# MAIN

def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx-model",    type=str, default="models/arabic-pii-detector-onnx")
    parser.add_argument("--test-file",     type=str, default="data/test.jsonl")
    parser.add_argument("--output-dir",    type=str, default="results_onnx")
    parser.add_argument("--latency-samples", type=int, default=200)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading ONNX model from: {args.onnx_model}")
    detector = ONNXPIIDetector(args.onnx_model)

    test_data = load_jsonl(args.test_file)
    print(f"Loaded {len(test_data)} test examples")

    #  F1 metrics 
    print("\n F1 Evaluation (ONNX INT8, CPU)")
    predictions = []
    for ex in test_data:
        try:
            result = detector.predict(ex["text"])
            predictions.append({"text": ex["text"], "entities": result["entities"]})
        except Exception as e:
            print(f"  ! Error: {e}")
            predictions.append({"text": ex["text"], "entities": []})

    metrics = compute_metrics(test_data, predictions)
    metrics_path = out_dir / "onnx_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {metrics_path}")
    print(f"  Overall F1: {metrics['overall']['f1']}")
    print(f"  Per-class:")
    for lbl, m in metrics["per_class"].items():
        print(f"    {lbl:25s} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} (n={m['support']})")

    #  Latency 
    print("\n Latency Benchmark (ONNX INT8, CPU) ")
    latency_texts = [ex["text"] for ex in test_data[:args.latency_samples]]
    latency = measure_latency(detector, latency_texts)
    latency_path = out_dir / "onnx_latency_report.json"
    with latency_path.open("w", encoding="utf-8") as f:
        json.dump(latency, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {latency_path}")
    print(f"  p50={latency['overall']['p50_ms']}ms  "
          f"p95={latency['overall']['p95_ms']}ms  "
          f"max={latency['overall']['max_ms']}ms")
    print(f"  Within 150ms target: {latency['within_target']}")

    #  Summary 
    summary = {
        "model": "ONNX INT8 quantized (CPU)",
        "test_size": len(test_data),
        "f1": metrics["overall"]["f1"],
        "precision": metrics["overall"]["precision"],
        "recall": metrics["overall"]["recall"],
        "latency_p50_ms": latency["overall"]["p50_ms"],
        "latency_p95_ms": latency["overall"]["p95_ms"],
        "within_150ms_target": latency["within_target"],
        "hardware": latency["hardware"],
    }
    summary_path = out_dir / "onnx_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
