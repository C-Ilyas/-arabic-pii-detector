"""
upload_models.py
"""
import os
from pathlib import Path
from huggingface_hub import login, HfApi
from transformers import AutoModelForTokenClassification, AutoTokenizer

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass


def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("Set HF_TOKEN in .env or as an environment variable.")
    login(token=token)

    api = HfApi()
    pytorch_repo = os.environ.get("HF_PYTORCH_REPO", "C-Ilyas/arabic-pii-detector")
    onnx_repo    = os.environ.get("HF_ONNX_REPO",    "C-Ilyas/arabic-pii-detector-onnx")

    # PyTorch model 
    print(f"Pushing PyTorch model to {pytorch_repo} ...")
    model = AutoModelForTokenClassification.from_pretrained("models/arabic-pii-detector")
    tokenizer = AutoTokenizer.from_pretrained("models/arabic-pii-detector")
    model.push_to_hub(pytorch_repo)
    tokenizer.push_to_hub(pytorch_repo)

    api.create_repo(pytorch_repo, repo_type="model", exist_ok=True)
    api.upload_file(
        path_or_fileobj="model_card.md",
        path_in_repo="README.md",
        repo_id=pytorch_repo,
        repo_type="model",
    )
    for f in [
        "results/before_finetuning_metrics.json",
        "results/after_finetuning_metrics.json",
        "results/latency_report.json",
    ]:
        api.upload_file(
            path_or_fileobj=f,
            path_in_repo=f,
            repo_id=pytorch_repo,
            repo_type="model",
        )

    # ONNX model 
    print(f"Pushing ONNX model to {onnx_repo} ...")
    api.create_repo(onnx_repo, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path="models/arabic-pii-detector-onnx",
        repo_id=onnx_repo,
        repo_type="model",
    )
    api.upload_file(
        path_or_fileobj="model_card_onnx.md",
        path_in_repo="README.md",
        repo_id=onnx_repo,
        repo_type="model",
    )


if __name__ == "__main__":
    main()
