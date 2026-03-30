"""
bert_pipeline.py
Xinyan's task: BERT fine-tuning for BBC News topic classification.

Uses the processed CSV files already committed to the repository:
    data/processed/train.csv
    data/processed/val.csv
    data/processed/test.csv
    data/processed/debug.csv

Default model:
    bert-base-uncased

Example usage:
    python src/bert_pipeline.py
    python src/bert_pipeline.py --use-debug --epochs 1 --max-length 128
"""

import argparse
import os
import random

import pandas as pd

try:
    import torch
    from torch.nn.utils import clip_grad_norm_
    from torch.optim import AdamW
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:
    raise ImportError(
        "Missing PyTorch dependency. Run: pip install -r requirements.txt"
    ) from exc

try:
    from transformers import (
        BertForSequenceClassification,
        BertTokenizerFast,
        get_linear_schedule_with_warmup,
    )
except ImportError as exc:
    raise ImportError(
        "Missing transformers dependency. Run: pip install -r requirements.txt"
    ) from exc

try:
    from src.data_preprocessing import ID_TO_LABEL, LABEL_MAP
except ImportError:
    from data_preprocessing import ID_TO_LABEL, LABEL_MAP


DEFAULT_MODEL_NAME = "bert-base-uncased"
DEFAULT_OUTPUT_DIR = "outputs/bert"


def get_repo_root() -> str:
    """Return the repository root based on this file's location."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_path(relative_path: str) -> str:
    """Resolve a repository-relative path to an absolute path."""
    return os.path.join(get_repo_root(), relative_path)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Prefer CUDA, then MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_processed_split(csv_relative_path: str) -> pd.DataFrame:
    """Load a processed split and validate the shared schema."""
    csv_path = resolve_path(csv_relative_path)
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Missing processed split: '{csv_relative_path}'")

    df = pd.read_csv(csv_path)
    required_columns = {"label", "label_id", "text"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"Split '{csv_relative_path}' is missing columns: {sorted(missing)}"
        )

    return df.reset_index(drop=True)


class BBCNewsBertDataset(Dataset):
    """PyTorch dataset for tokenizing BBC News text for BERT."""

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: BertTokenizerFast,
        max_length: int,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        encoded = self.tokenizer(
            str(row["text"]),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {key: value.squeeze(0) for key, value in encoded.items()}
        item["labels"] = torch.tensor(int(row["label_id"]), dtype=torch.long)
        return item


def make_dataloader(
    df: pd.DataFrame,
    tokenizer: BertTokenizerFast,
    batch_size: int,
    max_length: int,
    shuffle: bool,
) -> DataLoader:
    """Build a DataLoader for a processed CSV split."""
    dataset = BBCNewsBertDataset(df=df, tokenizer=tokenizer, max_length=max_length)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def train_one_epoch(
    model: BertForSequenceClassification,
    dataloader: DataLoader,
    optimizer: AdamW,
    scheduler,
    device: torch.device,
) -> float:
    """Run one training epoch and return average loss."""
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        batch = {key: value.to(device) for key, value in batch.items()}

        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / max(len(dataloader), 1)


@torch.no_grad()
def predict_split(
    model: BertForSequenceClassification,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[float, list[int], list[float]]:
    """Return average loss, predicted class ids, and prediction confidence."""
    model.eval()
    total_loss = 0.0
    predictions: list[int] = []
    confidences: list[float] = []

    for batch in dataloader:
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = model(**batch)
        probs = torch.softmax(outputs.logits, dim=-1)
        confidence, preds = torch.max(probs, dim=-1)

        total_loss += outputs.loss.item()
        predictions.extend(preds.cpu().tolist())
        confidences.extend(confidence.cpu().tolist())

    avg_loss = total_loss / max(len(dataloader), 1)
    return avg_loss, predictions, confidences


def save_prediction_csv(
    df: pd.DataFrame,
    predictions: list[int],
    confidences: list[float],
    output_path: str,
) -> None:
    """Save a teammate-friendly prediction file."""
    result = df.reset_index(drop=True).copy()
    result.insert(0, "row_id", result.index)
    result["pred_label_id"] = predictions
    result["pred_label"] = result["pred_label_id"].map(ID_TO_LABEL)
    result["pred_confidence"] = confidences
    result.to_csv(output_path, index=False)


def save_training_history(history: list[dict[str, float]], output_path: str) -> None:
    """Save per-epoch train/validation loss."""
    history_df = pd.DataFrame(history)
    history_df.to_csv(output_path, index=False)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune BERT on the processed BBC News dataset."
    )
    parser.add_argument("--train-path", default="data/processed/train.csv")
    parser.add_argument("--val-path", default="data/processed/val.csv")
    parser.add_argument("--test-path", default="data/processed/test.csv")
    parser.add_argument("--debug-path", default="data/processed/debug.csv")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--use-debug",
        action="store_true",
        help="Use debug.csv instead of train.csv for faster smoke-testing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")

    if "distilbert" in args.model_name.lower():
        raise ValueError("This pipeline supports BERT only. Please use a BERT model.")

    train_relative_path = args.debug_path if args.use_debug else args.train_path
    train_df = load_processed_split(train_relative_path)
    val_df = load_processed_split(args.val_path)
    test_df = load_processed_split(args.test_path)

    output_dir = resolve_path(args.output_dir)
    checkpoint_dir = os.path.join(output_dir, "best_model")
    os.makedirs(output_dir, exist_ok=True)

    device = get_device()
    print(f"[config] model={args.model_name}")
    print(f"[config] device={device}")
    print(f"[config] train_rows={len(train_df)}  val_rows={len(val_df)}  test_rows={len(test_df)}")
    print(f"[config] output_dir={args.output_dir}")

    tokenizer = BertTokenizerFast.from_pretrained(args.model_name)
    model = BertForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABEL_MAP),
        id2label=ID_TO_LABEL,
        label2id=LABEL_MAP,
    )
    model.to(device)

    train_loader = make_dataloader(
        df=train_df,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        shuffle=True,
    )
    val_loader = make_dataloader(
        df=val_df,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        shuffle=False,
    )
    test_loader = make_dataloader(
        df=test_df,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        shuffle=False,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    total_training_steps = max(len(train_loader) * args.epochs, 1)
    warmup_steps = int(total_training_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    history: list[dict[str, float]] = []
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )
        val_loss, _, _ = predict_split(
            model=model,
            dataloader=val_loader,
            device=device,
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )

        print(
            f"[epoch {epoch}/{args.epochs}] "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            print(f"[checkpoint] Saved best model to '{args.output_dir}/best_model'")

    save_training_history(
        history=history,
        output_path=os.path.join(output_dir, "training_history.csv"),
    )

    best_model = BertForSequenceClassification.from_pretrained(checkpoint_dir)
    best_model.to(device)

    val_loss, val_predictions, val_confidences = predict_split(
        model=best_model,
        dataloader=val_loader,
        device=device,
    )
    test_loss, test_predictions, test_confidences = predict_split(
        model=best_model,
        dataloader=test_loader,
        device=device,
    )

    save_prediction_csv(
        df=val_df,
        predictions=val_predictions,
        confidences=val_confidences,
        output_path=os.path.join(output_dir, "val_predictions.csv"),
    )
    save_prediction_csv(
        df=test_df,
        predictions=test_predictions,
        confidences=test_confidences,
        output_path=os.path.join(output_dir, "test_predictions.csv"),
    )

    print(f"[done] Best validation loss: {best_val_loss:.4f}")
    print(f"[done] Final validation loss: {val_loss:.4f}")
    print(f"[done] Final test loss: {test_loss:.4f}")
    print(f"[done] Wrote '{args.output_dir}/training_history.csv'")
    print(f"[done] Wrote '{args.output_dir}/val_predictions.csv'")
    print(f"[done] Wrote '{args.output_dir}/test_predictions.csv'")


if __name__ == "__main__":
    main()
