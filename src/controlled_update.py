"""Fine-tune Pythia on a small, PopQA-disjoint language-modeling dataset."""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Iterable

MODEL_NAME = "EleutherAI/pythia-160m"
DATASET_NAME = "roneneldan/TinyStories"
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "controlled_update"
    / "pythia-160m-tinystories"
)


def select_training_texts(
    records: Iterable[dict[str, object]],
    text_column: str,
    sample_size: int,
) -> list[str]:
    """Return the first sample_size non-empty text values from dataset records."""
    texts = []
    for record in records:
        value = record.get(text_column)
        if isinstance(value, str) and value.strip():
            texts.append(value.strip())
            if len(texts) == sample_size:
                break

    if len(texts) != sample_size:
        raise ValueError(
            f"Requested {sample_size} training examples, but found {len(texts)} "
            f"non-empty values in column {text_column!r}."
        )
    return texts


def validate_training_args(
    sample_size: int,
    batch_size: int,
    max_length: int,
    epochs: int,
    learning_rate: float,
) -> None:
    """Reject invalid or impractical training parameters before loading resources."""
    if sample_size < 1:
        raise ValueError("sample_size must be at least 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if max_length < 2:
        raise ValueError("max_length must be at least 2")
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be a finite positive number")


def load_training_texts(
    dataset_name: str,
    split: str,
    text_column: str,
    sample_size: int,
    seed: int,
) -> list[str]:
    """Stream and deterministically shuffle a bounded sample from a HF dataset."""
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, split=split, streaming=True)
    dataset = dataset.shuffle(seed=seed, buffer_size=max(sample_size * 10, 1000))
    return select_training_texts(dataset, text_column, sample_size)


def fine_tune(
    model_name: str,
    texts: list[str],
    output_dir: Path,
    batch_size: int,
    max_length: int,
    epochs: int,
    learning_rate: float,
    seed: int,
) -> None:
    """Run a small causal-language-model update and save the resulting checkpoint."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError("The tokenizer must define a pad token or an EOS token.")
    tokenizer.padding_side = "right"
    if tokenizer.eos_token:
        texts = [f"{text}{tokenizer.eos_token}" for text in texts]

    encoded = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )
    labels = encoded["input_ids"].clone()
    labels[encoded["attention_mask"] == 0] = -100
    training_data = TensorDataset(
        encoded["input_ids"],
        encoded["attention_mask"],
        labels,
    )
    generator = torch.Generator().manual_seed(seed)
    batches = DataLoader(
        training_data,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )

    model = AutoModelForCausalLM.from_pretrained(model_name).float().to(device)
    model.config.pad_token_id = tokenizer.pad_token_id
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        seen_examples = 0
        for input_ids, attention_mask, batch_labels in batches:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            loss = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=batch_labels,
            ).loss
            if loss is None or not torch.isfinite(loss):
                raise RuntimeError("Fine-tuning produced a non-finite loss.")
            loss.backward()
            optimizer.step()

            batch_count = input_ids.shape[0]
            total_loss += loss.item() * batch_count
            seen_examples += batch_count

        print(f"epoch {epoch + 1}/{epochs}: loss={total_loss / seen_examples:.4f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved updated model to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--dataset-name", default=DATASET_NAME)
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--sample-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    try:
        validate_training_args(
            args.sample_size,
            args.batch_size,
            args.max_length,
            args.epochs,
            args.learning_rate,
        )
    except ValueError as error:
        parser.error(str(error))

    texts = load_training_texts(
        args.dataset_name,
        args.split,
        args.text_column,
        args.sample_size,
        args.seed,
    )
    fine_tune(
        args.model_name,
        texts,
        args.output_dir,
        args.batch_size,
        args.max_length,
        args.epochs,
        args.learning_rate,
        args.seed,
    )


if __name__ == "__main__":
    main()
