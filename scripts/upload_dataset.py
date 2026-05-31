"""
upload_dataset.py
Uploads the synthetic Arabic PII dataset to Hugging Face Datasets Hub.
"""
import os
import json
from pathlib import Path
from datasets import Dataset, DatasetDict, Features, Sequence, Value
from huggingface_hub import login

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass


def load_jsonl(path: str):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def flatten_example(ex):
    """Convert entity dicts to parallel lists for HF Dataset format."""
    entities = ex.get("entities", [])
    return {
        "text": ex["text"],
        "entity_texts":  [e["text"]  for e in entities],
        "entity_labels": [e["label"] for e in entities],
        "entity_starts": [e["start"] for e in entities],
        "entity_ends":   [e["end"]   for e in entities],
    }


def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("Set HF_TOKEN in .env or as an environment variable.")
    login(token=token)

    repo_id = "C-Ilyas/arabic-pii-dataset"
    data_dir = Path("data")

    print("Loading splits...")
    splits = {
        "train":      load_jsonl(data_dir / "train.jsonl"),
        "validation": load_jsonl(data_dir / "validation.jsonl"),
        "test":       load_jsonl(data_dir / "test.jsonl"),
    }

    features = Features({
        "text":           Value("string"),
        "entity_texts":   Sequence(Value("string")),
        "entity_labels":  Sequence(Value("string")),
        "entity_starts":  Sequence(Value("int32")),
        "entity_ends":    Sequence(Value("int32")),
    })

    dataset_dict = DatasetDict({
        name: Dataset.from_list(
            [flatten_example(ex) for ex in examples],
            features=features,
        )
        for name, examples in splits.items()
    })

    print(f"Dataset summary:")
    for split, ds in dataset_dict.items():
        print(f"  {split}: {len(ds)} examples")

    print(f"\nPushing to https://huggingface.co/datasets/{repo_id} ...")
    dataset_dict.push_to_hub(
        repo_id,
        private=False,
        commit_message="Upload synthetic Arabic PII dataset",
    )
    print(f"Done: https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    main()
