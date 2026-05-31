"""
optimize_onnx.py


"""
import argparse
from pathlib import Path

from optimum.onnxruntime import ORTModelForTokenClassification, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/arabic-pii-detector")
    parser.add_argument("--output", type=str, default="models/arabic-pii-detector-onnx")
    args = parser.parse_args()

    src = Path(args.model)
    dst = Path(args.output)
    dst.mkdir(parents=True, exist_ok=True)

    print(f"Loading {src} and exporting to ONNX...")
    ort_model = ORTModelForTokenClassification.from_pretrained(src, export=True)
    tokenizer = AutoTokenizer.from_pretrained(src)
    ort_model.save_pretrained(dst)
    tokenizer.save_pretrained(dst)

    print("Applying INT8 dynamic quantization...")
    quantizer = ORTQuantizer.from_pretrained(dst)
    qconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=False)
    quantizer.quantize(save_dir=dst, quantization_config=qconfig)

    print(f"\nQuantized ONNX model saved to {dst}")
    print("To use it in inference, load with ORTModelForTokenClassification.from_pretrained()")


if __name__ == "__main__":
    main()
