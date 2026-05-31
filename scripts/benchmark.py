"""
benchmark.py

"""
import argparse
import json
import time
import platform
import statistics
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

from infer import PIIDetector


# DATA LOADING
def load_jsonl(path):
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


# METRICS

def entity_tuple(e: Dict) -> Tuple[int, int, str]:
    return (e["start"], e["end"], e["label"])


def compute_metrics(gold_examples: List[Dict], pred_examples: List[Dict]) -> Dict:
    """
    Compute entity-level precision/recall/F1 with exact match.

    """
    assert len(gold_examples) == len(pred_examples)

    # Exact match counters
    overall_tp = 0
    overall_fp = 0
    overall_fn = 0

    # Per-class counters
    per_class = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    # Partial match counters (any token-level overlap with correct label)
    partial_tp = 0

    # Error examples for analysis
    false_positives = []
    false_negatives = []
    boundary_errors = []
    wrong_labels = []

    for gold_ex, pred_ex in zip(gold_examples, pred_examples):
        text = gold_ex["text"]
        gold_set = {entity_tuple(e) for e in gold_ex["entities"]}
        pred_set = {entity_tuple(e) for e in pred_ex["entities"]}

        # Exact matches
        exact_matches = gold_set & pred_set
        overall_tp += len(exact_matches)
        for (_, _, lbl) in exact_matches:
            per_class[lbl]["tp"] += 1

        # False positives: predicted but not in gold (exact)
        for fp_ent in pred_set - gold_set:
            overall_fp += 1
            per_class[fp_ent[2]]["fp"] += 1

            # Categorize: boundary error, wrong label, or pure FP
            s, e, lbl = fp_ent
            categorized = False
            for gs, ge, gl in gold_set:
                # Overlapping span?
                if s < ge and e > gs:
                    if lbl == gl:
                        boundary_errors.append({
                            "text": text,
                            "gold": {"text": text[gs:ge], "label": gl, "start": gs, "end": ge},
                            "pred": {"text": text[s:e], "label": lbl, "start": s, "end": e},
                        })
                    else:
                        wrong_labels.append({
                            "text": text,
                            "gold": {"text": text[gs:ge], "label": gl, "start": gs, "end": ge},
                            "pred": {"text": text[s:e], "label": lbl, "start": s, "end": e},
                        })
                    categorized = True
                    break
            if not categorized:
                false_positives.append({
                    "text": text,
                    "pred": {"text": text[s:e], "label": lbl, "start": s, "end": e},
                })

        # False negatives: in gold but not predicted (exact)
        for fn_ent in gold_set - pred_set:
            overall_fn += 1
            per_class[fn_ent[2]]["fn"] += 1
            s, e, lbl = fn_ent
            # Only count as "pure" FN if no overlap with any prediction
            any_overlap = any(ps < e and pe > s for ps, pe, _ in pred_set)
            if not any_overlap:
                false_negatives.append({
                    "text": text,
                    "gold": {"text": text[s:e], "label": lbl, "start": s, "end": e},
                })

        # Partial match: any overlap with correct label
        for gs, ge, gl in gold_set:
            for ps, pe, pl in pred_set:
                if ps < ge and pe > gs and pl == gl:
                    partial_tp += 1
                    break

    def prf(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)}

    overall = prf(overall_tp, overall_fp, overall_fn)
    overall["exact_match_tp"] = overall_tp
    overall["exact_match_fp"] = overall_fp
    overall["exact_match_fn"] = overall_fn

    per_class_metrics = {}
    for lbl, c in per_class.items():
        per_class_metrics[lbl] = {**prf(c["tp"], c["fp"], c["fn"]),
                                  "support": c["tp"] + c["fn"]}

    total_gold = overall_tp + overall_fn
    partial_recall = partial_tp / total_gold if total_gold > 0 else 0.0

    return {
        "overall": overall,
        "per_class": per_class_metrics,
        "partial_match_recall": round(partial_recall, 4),
        "error_examples": {
            "false_positives": false_positives[:10],
            "false_negatives": false_negatives[:10],
            "boundary_errors": boundary_errors[:10],
            "wrong_labels": wrong_labels[:10],
        },
    }


# LATENCY MEASUREMENT

def measure_latency(detector: PIIDetector, texts: List[str], warmup: int = 5) -> Dict:
    """Measure full-pipeline latency across varying input lengths."""
    # Warmup
    for t in texts[:warmup]:
        detector.predict(t)

    times_by_length = defaultdict(list)
    all_times = []

    for text in texts:
        t0 = time.perf_counter()
        _ = detector.predict(text)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        all_times.append(elapsed_ms)
        # Bucket by approximate token count
        tok_count = len(text.split())
        if tok_count <= 25:
            bucket = "short (<=25 words)"
        elif tok_count <= 60:
            bucket = "medium (26-60 words)"
        else:
            bucket = "long (>60 words)"
        times_by_length[bucket].append(elapsed_ms)

    def summarize(arr):
        arr_sorted = sorted(arr)
        n = len(arr_sorted)
        return {
            "count": n,
            "mean_ms": round(statistics.mean(arr_sorted), 2),
            "p50_ms": round(arr_sorted[int(n * 0.50)], 2),
            "p95_ms": round(arr_sorted[min(n - 1, int(n * 0.95))], 2),
            "max_ms": round(max(arr_sorted), 2),
        }

    report = {
        "hardware": {
            "device": detector.device,
            "torch_threads": torch.get_num_threads(),
            "platform": platform.platform(),
            "processor": platform.processor() or "unknown",
            "cuda_available": torch.cuda.is_available(),
        },
        "batch_size": 1,
        "max_input_length": max(len(t) for t in texts),
        "overall": summarize(all_times),
        "by_length": {k: summarize(v) for k, v in times_by_length.items()},
        "target_ms": 150,
        "within_target": summarize(all_times)["p95_ms"] < 150,
    }
    return report


# MAIN
def predict_all(detector: PIIDetector, examples: List[Dict]) -> List[Dict]:
    """Run inference on every example, return entity lists."""
    predictions = []
    for ex in examples:
        try:
            result = detector.predict(ex["text"])
            predictions.append({"text": ex["text"], "entities": result["entities"]})
        except Exception as e:
            print(f"  ! Error on example: {e}")
            predictions.append({"text": ex["text"], "entities": []})
    return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-file", type=str, default="data/test.jsonl")
    parser.add_argument("--base-model", type=str,
                        default="aubmindlab/bert-base-arabertv02",
                        help="Base model (no fine-tuning) for before comparison")
    parser.add_argument("--finetuned-model", type=str,
                        default="models/arabic-pii-detector")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--latency-samples", type=int, default=200)
    parser.add_argument("--skip-base", action="store_true",
                        help="Skip base model eval (base has no PII heads, will be ~0)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    test_data = load_jsonl(args.test_file)
    print(f"Loaded {len(test_data)} test examples")

    # AFTER fine-tuning 
    print(f"\n=== Evaluating fine-tuned model: {args.finetuned_model} ===")
    ft_detector = PIIDetector(args.finetuned_model)
    ft_preds = predict_all(ft_detector, test_data)
    ft_metrics = compute_metrics(test_data, ft_preds)
    ft_path = out_dir / "after_finetuning_metrics.json"
    with ft_path.open("w", encoding="utf-8") as f:
        json.dump(ft_metrics, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {ft_path}")
    print(f"  Overall F1: {ft_metrics['overall']['f1']}")
    print(f"  Per-class:")
    for lbl, m in ft_metrics["per_class"].items():
        print(f"    {lbl:25s} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} (n={m['support']})")

    # LATENCY 
    print(f"\n=== Latency benchmark ===")
    latency_texts = [ex["text"] for ex in test_data[:args.latency_samples]]
    latency_report = measure_latency(ft_detector, latency_texts)
    lat_path = out_dir / "latency_report.json"
    with lat_path.open("w", encoding="utf-8") as f:
        json.dump(latency_report, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {lat_path}")
    print(f"  Overall: p50={latency_report['overall']['p50_ms']}ms, "
          f"p95={latency_report['overall']['p95_ms']}ms, "
          f"max={latency_report['overall']['max_ms']}ms")
    print(f"  Within 150ms target: {latency_report['within_target']}")

    #  BEFORE fine-tuning 
    if not args.skip_base:
        print(f"\n=== Evaluating base model (no fine-tuning): {args.base_model} ===")
        print("  NOTE: Base model has no PII-specific head; expect near-zero F1.")
        print("  This is the deliberate 'before' baseline to show fine-tuning impact.")
        try:

            import torch.nn as nn
            base_tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
            # Build base model directly from base config + random classification head
            base_model = AutoModelForTokenClassification.from_pretrained(
                args.base_model,
                num_labels=len(ft_detector.id2label),
                id2label=ft_detector.id2label,
                label2id={v: k for k, v in ft_detector.id2label.items()},
                ignore_mismatched_sizes=True,
            )
            # Re initialise classifier head to random weights (simulate "before" state)
            nn.init.normal_(base_model.classifier.weight, std=0.02)
            nn.init.zeros_(base_model.classifier.bias)

            class BaseDetector(PIIDetector):
                def __init__(self, tok, mdl, dev):
                    self.tokenizer = tok
                    self.model = mdl.to(dev)
                    self.model.eval()
                    self.device = dev
                    self.id2label = mdl.config.id2label

            base_detector = BaseDetector(base_tokenizer, base_model, ft_detector.device)
            base_preds = predict_all(base_detector, test_data)
            base_metrics = compute_metrics(test_data, base_preds)
            base_path = out_dir / "before_finetuning_metrics.json"
            with base_path.open("w", encoding="utf-8") as f:
                json.dump(base_metrics, f, ensure_ascii=False, indent=2)
            print(f"  Saved: {base_path}")
            print(f"  Overall F1: {base_metrics['overall']['f1']}")
        except Exception as e:
            print(f"  Could not evaluate base model: {e}")

    #  COMPARISON SUMMARY 
    summary = {
        "test_size": len(test_data),
        "before_f1": None,
        "after_f1": ft_metrics["overall"]["f1"],
        "latency_p95_ms": latency_report["overall"]["p95_ms"],
        "within_150ms_target": latency_report["within_target"],
    }
    base_path = out_dir / "before_finetuning_metrics.json"
    if base_path.exists():
        with base_path.open() as f:
            summary["before_f1"] = json.load(f)["overall"]["f1"]
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
