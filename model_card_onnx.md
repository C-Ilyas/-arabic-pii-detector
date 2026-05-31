---
language:
  - ar
  - en
tags:
  - token-classification
  - ner
  - pii
  - arabic
  - arabic-dialect
  - algeria
  - privacy
  - onnx
  - int8
license: apache-2.0
base_model: aubmindlab/bert-base-arabertv02
pipeline_tag: token-classification
---

# Arabic PII Detector — ONNX INT8 (CPU Optimized)

This is the **ONNX INT8 quantized** version of [C-Ilyas/arabic-pii-detector](https://huggingface.co/C-Ilyas/arabic-pii-detector), optimized for CPU deployment.

The model was exported from PyTorch to ONNX format and dynamically quantized to INT8 using AVX2 instructions — matched to the AMD Ryzen 5 3500U instruction set. It achieves **identical F1 to the PyTorch model** with significantly lower CPU latency.

## Performance

| Metric | PyTorch (T4 GPU) | ONNX INT8 (Ryzen 5 CPU) |
|---|---|---|
| Overall F1 | 0.9916 | **0.9916** |
| Latency p50 | 7.35ms | 57.29ms |
| Latency p95 | 7.97ms | **96.38ms** |
| Within 150ms target | ✓ | ✓ |

**Zero accuracy loss** from quantization — F1 preserved at 0.9916.

**Hardware**: CPU benchmark on ThinkPad AMD Ryzen 5 3500U (4 cores, no discrete GPU).

## Detected Entity Types

`PERSON`, `EMAIL`, `PHONE_NUMBER`, `ADDRESS`, `ACCOUNT_NUMBER`, `BANK_ACCOUNT_NUMBER`, `IBAN`

## Usage

```python
from optimum.onnxruntime import ORTModelForTokenClassification
from transformers import AutoTokenizer

model = ORTModelForTokenClassification.from_pretrained("C-Ilyas/arabic-pii-detector-onnx")
tokenizer = AutoTokenizer.from_pretrained("C-Ilyas/arabic-pii-detector-onnx")
```

For the full inference pipeline (BIO decoding, redaction, confidence scores) see `scripts/infer_onnx.py` in the [project repository](https://huggingface.co/C-Ilyas/arabic-pii-detector).

## How it was created

```bash
python scripts/optimize_onnx.py \
    --model models/arabic-pii-detector \
    --output models/arabic-pii-detector-onnx
```

Uses `AutoQuantizationConfig.avx2(is_static=False, per_channel=False)` — dynamic INT8 quantization with AVX2 kernels.

## PyTorch Model

The original PyTorch model with full documentation, training details, and per-class metrics is at:
**[C-Ilyas/arabic-pii-detector](https://huggingface.co/C-Ilyas/arabic-pii-detector)**
