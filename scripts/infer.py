"""
infer.py
Runs inference on input text, returning redacted text + structured entity list.

"""
import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification


# REDACTION TAGS (matches assessment spec)
LABEL_TO_TAG = {
    "PERSON": "[PERSON]",
    "EMAIL": "[EMAIL]",
    "PHONE_NUMBER": "[PHONE_NUMBER]",
    "ADDRESS": "[ADDRESS]",
    "ACCOUNT_NUMBER": "[ACCOUNT_NUMBER]",
    "BANK_ACCOUNT_NUMBER": "[BANK_ACCOUNT_NUMBER]",
    "IBAN": "[IBAN]",
}


# REGEX HELPERS 
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")


def _iban_mod97_check(iban: str) -> bool:
    iban = iban.replace(" ", "").upper()
    if len(iban) < 15:
        return False
    rearranged = iban[4:] + iban[:4]
    digits = ""
    for ch in rearranged:
        if ch.isdigit():
            digits += ch
        elif ch.isalpha():
            digits += str(ord(ch) - 55)
        else:
            return False
    rem = 0
    for d in digits:
        rem = (rem * 10 + int(d)) % 97
    return rem == 1


def validate_entity(text: str, label: str) -> float:
    """Return a confidence multiplier (1.0 = no change, <1 = penalty, >1 = boost)."""
    if label == "EMAIL":
        return 1.05 if EMAIL_RE.fullmatch(text.strip()) else 0.85
    if label == "IBAN":
        if _iban_mod97_check(text.strip()):
            return 1.05
        return 0.80
    return 1.0


# DIGIT NORMALIZATION
ARABIC_INDIC = "٠١٢٣٤٥٦٧٨٩"
WESTERN = "0123456789"
NORMALIZE_DIGITS_MAP = str.maketrans(ARABIC_INDIC, WESTERN)


def normalize_for_model(text: str) -> Tuple[str, List[int]]:
    """
    Normalize text for the model while preserving a char-by-char mapping back
    to the ORIGINAL text. Since digit normalization is 1:1 char, mapping is identity.
    Returns (normalized_text, idx_map) where idx_map[i] = original index of char i.

    We keep this 1:1 to make offset mapping trivial. If you later add many-to-one
    normalizations (e.g. removing kashida), update this function and downstream
    offset handling will Just Work.
    """
    normalized = text.translate(NORMALIZE_DIGITS_MAP)
    idx_map = list(range(len(text)))
    assert len(normalized) == len(text), "Normalization must be 1:1 for offset safety"
    return normalized, idx_map



# BIO DECODING
def bio_decode_to_spans(
    tokens_info: List[Tuple[int, int, str, float]],
) -> List[Dict]:
    """
    Convert per-token (start, end, label, prob) into entity spans.

    Handles:
    - B-X starts a new entity
    - I-X continues if matching previous label, else treats as B-X (lenient)
    - O ends current entity
    """
    entities = []
    current = None  

    for (s, e, label, prob) in tokens_info:
        if label == "O" or label.startswith("-"):
            if current is not None:
                entities.append(current)
                current = None
            continue

        prefix, etype = label[0], label[2:]

        if prefix == "B" or current is None or current["label"] != etype:
            if current is not None:
                entities.append(current)
            current = {"label": etype, "start": s, "end": e, "probs": [prob]}
        else:
            current["end"] = e
            current["probs"].append(prob)

    if current is not None:
        entities.append(current)

    # Average probability as confidence
    for ent in entities:
        ent["confidence"] = float(np.mean(ent["probs"]))
        del ent["probs"]

    return entities


# MAIN INFERENCE
class PIIDetector:
    def __init__(self, model_path: str, device: Optional[str] = None, onnx: bool = False):
        self.onnx = onnx
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        if onnx:
            # ONNX runtime is CPU-only here; lazy import so torch-only users don't need optimum.
            from optimum.onnxruntime import ORTModelForTokenClassification
            self.model = ORTModelForTokenClassification.from_pretrained(model_path)
            self.device = "cpu"
        else:
            self.model = AutoModelForTokenClassification.from_pretrained(model_path)
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = device
            self.model.to(device)
            self.model.eval()
        self.id2label = self.model.config.id2label

    @torch.inference_mode()
    def predict(self, text: str, max_length: int = 256) -> Dict:
        #  Normalize (digits only, 1:1 mapping preserved)
        normalized, _idx_map = normalize_for_model(text)

        #  Tokenize with offset mapping
        enc = self.tokenizer(
            normalized,
            return_offsets_mapping=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        offsets = enc.pop("offset_mapping")[0].tolist()
        enc = {k: v.to(self.device) for k, v in enc.items()}

        #  Forward pass
        logits = self.model(**enc).logits[0]  # (seq_len, num_labels)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        pred_ids = probs.argmax(axis=-1)
        pred_labels = [self.id2label[int(p)] for p in pred_ids]
        max_probs = probs.max(axis=-1)

        #  Build per-token info, skipping special tokens
        tokens_info = []
        for i, (s, e) in enumerate(offsets):
            if s == 0 and e == 0:
                continue
            tokens_info.append((s, e, pred_labels[i], float(max_probs[i])))

        #  BIO decode : entity spans (in normalized text, which == original positions)
        raw_entities = bio_decode_to_spans(tokens_info)

        # Build final entity records with text slice from ORIGINAL string
        entities = []
        for ent in raw_entities:
            start, end = ent["start"], ent["end"]
            ent_text = text[start:end]
            # Trim trailing whitespace or punctuation that's clearly not PII
            ent_text_stripped = ent_text.rstrip(" .،,;:")
            if len(ent_text_stripped) < len(ent_text):
                end -= (len(ent_text) - len(ent_text_stripped))
                ent_text = ent_text_stripped

            if not ent_text:
                continue

            # Apply regex validation as confidence modifier
            conf = ent["confidence"] * validate_entity(ent_text, ent["label"])
            conf = min(0.999, max(0.0, conf))

            entities.append({
                "text": ent_text,
                "label": ent["label"],
                "start": start,
                "end": end,
                "confidence": round(conf, 4),
            })

        #  Build redacted text (reverse order to keep offsets valid)
        redacted = text
        for ent in sorted(entities, key=lambda x: -x["start"]):
            tag = LABEL_TO_TAG.get(ent["label"], f"[{ent['label']}]")
            redacted = redacted[:ent["start"]] + tag + redacted[ent["end"]:]

        return {
            "redacted_text": redacted,
            "entities": entities,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, default=None, help="Text to analyze")
    parser.add_argument("--file", type=str, default=None, help="Read text from file")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to fine-tuned model (defaults to the PyTorch or ONNX "
                             "model dir depending on --onnx)")
    parser.add_argument("--onnx", action="store_true",
                        help="Use the ONNX INT8 model in models/arabic-pii-detector-onnx (CPU)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    args = parser.parse_args()

    # Default model path depends on the backend.
    if args.model is None:
        args.model = ("models/arabic-pii-detector-onnx" if args.onnx
                      else "models/arabic-pii-detector")

    # Get input text
    if args.text:
        text = args.text
    elif args.file:
        text = Path(args.file).read_text(encoding="utf-8")
    else:
        text = sys.stdin.read()

    if not text.strip():
        print(json.dumps({"redacted_text": "", "entities": []}, ensure_ascii=False))
        return

    detector = PIIDetector(args.model, device=args.device, onnx=args.onnx)
    result = detector.predict(text, max_length=args.max_length)

    if args.pretty:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
