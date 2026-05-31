# Arabic PII Detection

Fine-tuned BERT-based model that detects PII (Personally Identifiable Information) in Arabic, English, and mixed Arabic/English text, then returns redacted text alongside structured entity spans with confidence scores. Covers MSA, Gulf, Egyptian, and Algerian dialect phrasings.

## Project Overview

- **Task**: Token classification (NER) over 7 PII categories
- **Base model**: `aubmindlab/bert-base-arabertv02`
- **Approach**: Fine-tuned with BIO tagging on synthetic Arabic data (50K examples)
- **Training hardware**: NVIDIA Tesla T4 GPU (Google Colab)
- **Latency target**: <150ms full pipeline per input
- **Val F1**: 0.9682 (held-out unseen templates) | **Test F1**: 0.9916 | Latency p95: 7.97ms
- **Output format**: JSON with `redacted_text` + `entities[]` (each with text, label, start, end, confidence)

## Detected Entities

`PERSON`, `EMAIL`, `PHONE_NUMBER`, `ADDRESS`, `ACCOUNT_NUMBER`, `BANK_ACCOUNT_NUMBER`, `IBAN`

## Project Structure

```
.
├── README.md
├── requirements.txt
├── data/
│   ├── train.jsonl          # 50,000 examples
│   ├── validation.jsonl     # 2,000 examples (held-out templates)
│   └── test.jsonl           # 2,000 examples (held-out templates)
├── scripts/
│   ├── entity_pools.py      # Generators for each PII type
│   ├── prepare_data.py      # Synthetic data generation
│   ├── train.py             # Fine-tuning entry point
│   ├── infer.py             # Inference CLI / library
│   ├── benchmark.py         # Before/after eval + latency
│   ├── benchmark_onnx.py    # CPU ONNX benchmark
│   └── optimize_onnx.py     # ONNX + INT8 quantization
├── configs/
│   └── training_config.yaml
├── results/
│   ├── before_finetuning_metrics.json
│   ├── after_finetuning_metrics.json
│   └── latency_report.json
└── model_card.md
```

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate synthetic dataset
python scripts/prepare_data.py

# 3. Fine-tune (uses configs/training_config.yaml)
python scripts/train.py

# 4. Run inference on a sentence
python scripts/infer.py --text "اسمي محمد أحمد ورقم تليفوني 01012345678" --pretty

# 5. Benchmark base vs fine-tuned + measure latency
python scripts/benchmark.py
```

## Inference Examples

**Example 1** — Arabic with PERSON and PHONE:

```bash
python scripts/infer.py --text "اسمي محمد أحمد ورقم تليفوني 01012345678" --pretty
```

```json
{
  "redacted_text": "اسمي [PERSON] ورقم تليفوني [PHONE_NUMBER]",
  "entities": [
    {"text": "محمد أحمد", "label": "PERSON", "start": 5, "end": 14, "confidence": 0.96},
    {"text": "01012345678", "label": "PHONE_NUMBER", "start": 28, "end": 39, "confidence": 0.99}
  ]
}
```

**Example 2** — Algerian dialect:

```bash
python scripts/infer.py --text "راني سفيان بوزيد ورقمي 0698123456 من الجزائر العاصمة" --pretty
```

**Example 3** — Arabic with IBAN:

```bash
python scripts/infer.py --text "رقم الآيبان هو DZ4400799000001234567890" --pretty
```

**Example 4** — Address with Arabic-Indic digits:

```bash
python scripts/infer.py --text "العنوان: ١٢ شارع التحرير، الدقي، الجيزة" --pretty
```

## Dataset Creation

Data is generated **synthetically** by `scripts/prepare_data.py` using:

1. **Template-based generation** (75% of examples): ~130 Arabic + mixed sentence templates covering single-entity, multi-entity, code-switched, and Algerian dialect cases.
2. **Realistic value pools** (`scripts/entity_pools.py`):

   - **Names**: Arabic (MSA + Gulf + Egyptian + Algerian) + Berber-influenced names + French-influenced transliterations (Yazid, Sofiane, Benali, Zerrouki…)
   - **Phones**: Egyptian (010/011/012/015), Saudi (+966 5XX), UAE (+971 5X), **Algerian (+213 5XX/6XX/7XX or local 05/06/07)**, random spacing/dashes, 30% Arabic Indic digit conversion
   - **IBANs**: Valid mod-97 checksums for EG/SA/AE/JO/KW/QA/BH/**DZ**
   - **Emails**: Realistic user@domain , includes `.dz`, `.fr`, `yahoo.fr`, `hotmail.fr`, `univ-alger.dz`
   - **Addresses**: 20 Algerian cities (الجزائر، وهران، قسنطينة، تيزي وزو، تمنراست…), Algerian districts and street names, plus Egypt, Saudi Arabia, UAE
   - **Accounts**: short alphanumeric (ACCOUNT_NUMBER) vs longer numeric (BANK_ACCOUNT_NUMBER)
3. **Negative examples** (25%): sentences with no PII, including hard distractors that resemble PII ,Algerian tax IDs (NIF), SWIFT codes, IMEI numbers, decree numbers, invoice refs, postal tracking codes, customs codes.

**Critical detail**: character offsets are computed from the FINAL string, guaranteeing exact alignment even with RTL text and mixed scripts. Every offset is asserted at generation time.

**Evaluation split strategy**: All three splits use **structurally distinct held-out templates** , train, val, and test each have unique sentence patterns never seen in the other splits. Both val and test templates are designed with equal difficulty: a mix of simple single-entity, ambiguous mid-sentence, and hard multi-entity (up to 4 entities) cases, including the `ACCOUNT_NUMBER` vs `BANK_ACCOUNT_NUMBER` confusion pair. This ensures val F1 during training and test F1 during benchmarking are directly comparable and neither is inflated.

| Split      | Examples | Templates                          |
| ---------- | -------- | ---------------------------------- |
| train      | 50,000   | ~110 (from TEMPLATES pool)         |
| validation | 2,000    | 21 (VAL_ONLY_TEMPLATES, held-out)  |
| test       | 2,000    | 21 (TEST_ONLY_TEMPLATES, held-out) |

## Model Choice

`aubmindlab/bert-base-arabertv02` was selected after evaluating alternatives:

| Criterion         | AraBERT-base (chosen) | AraBERT-large       | CAMeLBERT-mix | XLM-RoBERTa-large |
| ----------------- | --------------------- | ------------------- | ------------- | ----------------- |
| Arabic coverage   | MSA + large corpora   | MSA + large corpora | MSA + DA + CA | Multilingual      |
| Parameters        | ~135M                 | ~340M               | ~135M         | ~550M             |
| GPU inference     | ✓                    | ✓                  | ✓            | ✓                |
| CPU ONNX INT8 p95 | ~60–90ms ✓          | ~200–300ms ✗      | ~60–80ms ✓  | too slow ✗       |
| NER track record  | ✓                    | ✓                  | ✓            | ✓                |
| Safetensors       | ✓                    | ✓                  | ✗            | ✓                |

**Why not bert-large?** During experimentation, `aubmindlab/bert-large-arabertv02` (340M params) achieved F1=0.9581 but failed the CPU latency target (p95=239ms with ONNX INT8). bert-base achieves  p95=7.97ms on GPU and  ~60–90ms on CPU with ONNX INT8 — comfortably within the 150ms budget.

**Why not mdeberta-v3-base?** DeBERTa's disentangled attention adds CPU overhead , estimated p95 ~150–250ms, borderline at best. Not worth the complexity for marginal F1 gain.

**Why not XLM-RoBERTa-large?** 550M parameters , too slow for CPU deployment.

**LLM-based approaches** (zero-shot GPT-4 etc.) were ruled out: the <150ms latency budget makes them infeasible regardless of accuracy.

## Fine-tuning Approach

- **Task framing**: Token classification with BIO tagging (15 labels: `O` + 7 × `{B-, I-}`)
- **Why BIO over seq2seq**: Single forward pass, exact spans for free, faster training, more interpretable
- **Subword alignment**: First subword of each entity span gets `B-LABEL`; subsequent subwords get `I-LABEL`; everything else gets `O`. Special tokens get `-100` (ignored by loss)
- **Loss**: CrossEntropy, `load_best_model_at_end` on val F1

### Hyperparameters and Reasoning

| Param                   | Value            | Why                                                                                                                                                                                     |
| ----------------------- | ---------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Epochs                  | 12 (early stop)  | Ceiling ,early stopping finds the peak automatically                                                                                                                                    |
| Early stopping patience | 3                | Cosine schedule can dip one epoch before recovering; patience=3 avoids stopping too early                                                                                               |
| Batch size              | 32               | bert-base fits larger batches; more stable gradients than batch=16                                                                                                                      |
| Learning rate           | 3e-5             | bert-base needs a higher lr than bert-large , more capacity per layer means faster convergence                                                                                          |
| Scheduler               | **Cosine** | Cosine stays near peak lr for longer then decays smoothly , consistently outperforms linear on NER by 0.5–1.5 F1 points. Linear decays too aggressively in the productive middle phase |
| Warmup ratio            | 0.06             | Shorter warmup with 50K examples , over-warming wastes productive training steps                                                                                                        |
| Weight decay            | 0.01             | Standard for bert-base ,0.1 (used for bert-large) was too aggressive and prevented the smaller model from fitting well                                                                  |
| Label smoothing         | 0.0              | bert-base is underconfident during training; label smoothing makes it even less decisive, hurting recall                                                                                |
| Max length              | 256              |                                                                                                                                                                                         |
| Optimizer               | AdamW            |                                                                                                                                                                                         |
| fp16                    | true (GPU)       | Half-precision halves memory, speeds up ~2× on T4                                                                                                                                      |
| Seed                    | 42               |                                                                                                                                                                                         |

**Key insight**: the hyperparameters were deliberately different from the bert-large run. bert-large needed heavy regularization (weight_decay=0.1, dropout=0.2, label_smoothing=0.1) to prevent overfitting its 340M parameters on synthetic data. bert-base (135M params) needs lighter regularization and a higher learning rate , applying bert-large settings to bert-base caused underfit and early stopping at epoch 2 with F1=0.90.

## Inference Pipeline

```
input text
    ↓
Digit normalization (٠١٢٣ → 0123, 1:1 mapping preserves offsets)
    ↓
Tokenize with offset_mapping=True
    ↓
Model forward pass → logits → softmax + argmax
    ↓
BIO decode → entity spans (offsets reference ORIGINAL text)
    ↓
Regex validation layer (helper only):
    - IBAN mod-97 checksum check (boost or penalty)
    - Email format check
    ↓
Trim trailing whitespace / punctuation from spans
    ↓
Build redacted text (reverse-order replacement to preserve offsets)
    ↓
JSON output
```

## Benchmarking

Run with:

```bash
python scripts/benchmark.py
```

Produces:

- `results/before_finetuning_metrics.json` — base AraBERT-base (random PII head) on test set
- `results/after_finetuning_metrics.json` — fine-tuned model on test set
- `results/latency_report.json` — full-pipeline latency profile

Metrics computed:

- Overall entity-level precision, recall, F1 (exact match)
- Per-class P/R/F1 for all 7 entity types
- Partial match recall
- Error analysis: false positives, false negatives, boundary errors, label confusion
- Latency: p50, p95, max, mean by input length bucket

### Results

**Training & benchmark hardware**: NVIDIA Tesla T4 GPU (Google Colab)

|                      | Before fine-tuning | After fine-tuning |
| -------------------- | ------------------ | ----------------- |
| Overall F1           | 0.0006             | **0.9916**  |
| Latency p95 (T4 GPU) | —                 | 7.97ms            |
| Latency p50 (T4 GPU) | —                 | 7.35ms            |
| Within 150ms         | —                 | ✓                |

| Class               | Precision | Recall | F1    | Support |
| ------------------- | --------- | ------ | ----- | ------- |
| EMAIL               | 1.000     | 1.000  | 1.000 | 283     |
| PERSON              | 0.996     | 0.996  | 0.996 | 516     |
| IBAN                | 0.991     | 1.000  | 0.995 | 317     |
| PHONE_NUMBER        | 0.994     | 0.991  | 0.993 | 350     |
| BANK_ACCOUNT_NUMBER | 0.968     | 1.000  | 0.984 | 273     |
| ACCOUNT_NUMBER      | 0.990     | 0.958  | 0.974 | 214     |
| ADDRESS             | 0.990     | 0.990  | 0.990 | 296     |

**Val F1** (held-out, used for early stopping): **0.9682**
**Test F1** (held-out, final evaluation): **0.9916**

> The 2.3-point gap between val and test F1 is expected: both use structurally distinct templates of equal difficulty, but random variation in generated entity values and template sampling means some test batches are slightly easier. In real-world deployment expect performance closer to val F1 (0.9682) since real text contains noise not present in synthetic data.

## Latency Optimization

To hit <150ms on CPU, the inference pipeline:

- Uses `torch.inference_mode()` to skip autograd
- Tokenizes with fast (Rust-based) tokenizer
- ONNX export + INT8 dynamic quantization via `scripts/optimize_onnx.py`

```bash
# Export to ONNX + INT8 quantization (AVX2 config for AMD/Intel CPUs without AVX-512)
python scripts/optimize_onnx.py --model models/arabic-pii-detector

# Benchmark ONNX model on CPU
python scripts/benchmark_onnx.py
```

### CPU Benchmark Results (ONNX INT8, AVX2)

Two separate hardware environments were used:

- **Training + GPU benchmark**: NVIDIA Tesla T4 GPU (Google Colab) , fast matrix ops, FP16 training
- **CPU benchmark**: ThinkPad with AMD Ryzen 5 3500U (4 cores, integrated Radeon Vega GPU, no discrete GPU)  , represents a typical CPU-only deployment machine

The AVX2 quantization config was chosen specifically for the Ryzen 5 3500U, which supports AVX2 but not AVX-512 VNNI. Using the wrong instruction set (AVX-512) would silently fall back to a slower path.

| Metric       | PyTorch (T4 GPU) | ONNX INT8 AVX2 (Ryzen 5 3500U CPU) |
| ------------ | ---------------- | ---------------------------------- |
| F1           | 0.9916           | **0.9916**                   |
| Precision    | —               | 0.9907                             |
| Recall       | —               | 0.9924                             |
| p50 latency  | 7.35ms           | 57.29ms                            |
| p95 latency  | 7.97ms           | **96.38ms**                  |
| Within 150ms | ✓               | ✓                                 |

**Key result**: ONNX INT8 quantization preserves full accuracy (F1 unchanged at 0.9916) while achieving **96.38ms p95 on CPU** , well within the 150ms target. This confirms bert-base is the right choice for CPU-deployable Arabic PII detection.

## Reproduction

```bash
pip install -r requirements.txt
python scripts/prepare_data.py
python scripts/train.py
python scripts/benchmark.py
```

The seed is fixed (`42`), so dataset generation and training are reproducible.

## Known Limitations

- **Synthetic data domain gap**: real-world Arabic text has OCR errors, dialectal variations, and structural patterns not covered by templates
- **ACCOUNT_NUMBER vs BANK_ACCOUNT_NUMBER**: distinguished by length and context ,borderline-length numeric strings may be confused
- **No PII types beyond the 7 listed**: national IDs, passport numbers, credit cards are NOT detected
- **Maximum input length**: 256 tokens (longer inputs truncated)

## Possible Improvements

1. Real data augmentation from anonymized public Arabic NER corpora (e.g. AQMAR, ANERcorp)
2. LLM-augmented template diversification for richer dialect and domain coverage
3. CRF layer on top of token classifier for cleaner span boundaries
4. Noisy text augmentation (typos, missing diacritics, inconsistent spacing) to close the synthetic-to-real gap
5. Confidence threshold tuning in `infer.py` to reduce low-confidence false positives
6. Separate model for ACCOUNT_NUMBER vs BANK_ACCOUNT_NUMBER

## Deliverables Checklist

- [X] Hugging Face model: `C-Ilyas/arabic-pii-detector`
- [X] Code repository (this repo)
- [X] Dataset for train/val/test (regenerated by `prepare_data.py`)
- [X] `prepare_data.py`, `train.py`, `infer.py`, `benchmark.py`, `benchmark_onnx.py`
- [X] `before_finetuning_metrics.json` + `after_finetuning_metrics.json` + `latency_report.json`
- [X] README + model_card.md
- [X] Clear reproduction commands
