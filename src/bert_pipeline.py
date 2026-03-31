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

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

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


def resolve_path(path: str) -> str:
    """Resolve absolute paths directly and relative paths from the repo root."""
    path = os.path.expanduser(path)
    if os.path.isabs(path):
        return os.path.abspath(path)
    return os.path.abspath(os.path.join(get_repo_root(), path))


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
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

    return validate_processed_split(df=df, split_name=csv_relative_path)


def validate_processed_split(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """Validate processed split contents against the shared schema."""
    df = df.reset_index(drop=True).copy()

    if df.empty:
        raise ValueError(f"Split '{split_name}' is empty.")

    if df[["label", "label_id", "text"]].isna().any().any():
        raise ValueError(f"Split '{split_name}' contains missing label/text values.")

    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["text"] = df["text"].astype(str)

    empty_text = df["text"].str.strip().eq("")
    if empty_text.any():
        raise ValueError(f"Split '{split_name}' contains empty text rows.")

    label_id_numeric = pd.to_numeric(df["label_id"], errors="coerce")
    if label_id_numeric.isna().any():
        raise ValueError(f"Split '{split_name}' has non-numeric label_id values.")

    label_id_array = label_id_numeric.to_numpy()
    label_id_int = label_id_numeric.astype(int)
    if not np.array_equal(label_id_array, label_id_int.to_numpy()):
        raise ValueError(f"Split '{split_name}' has non-integer label_id values.")

    df["label_id"] = label_id_int

    valid_labels = set(LABEL_MAP)
    invalid_labels = sorted(set(df["label"]) - valid_labels)
    if invalid_labels:
        raise ValueError(
            f"Split '{split_name}' contains unknown labels: {invalid_labels}"
        )

    valid_label_ids = set(LABEL_MAP.values())
    invalid_label_ids = sorted(set(df["label_id"]) - valid_label_ids)
    if invalid_label_ids:
        raise ValueError(
            f"Split '{split_name}' contains invalid label_id values: {invalid_label_ids}"
        )

    expected_label_ids = df["label"].map(LABEL_MAP)
    mismatch_mask = df["label_id"] != expected_label_ids
    if mismatch_mask.any():
        raise ValueError(
            f"Split '{split_name}' has {int(mismatch_mask.sum())} label/label_id mismatches."
        )

    return df


def validate_args(args: argparse.Namespace) -> None:
    """Validate command-line arguments before training starts."""
    if not args.model_name.strip():
        raise ValueError("--model-name must not be empty.")

    model_name = args.model_name.lower()
    if "distilbert" in model_name:
        raise ValueError("This pipeline supports BERT only. Please use a BERT model.")
    if "bert" not in model_name:
        raise ValueError("--model-name must point to a BERT model.")

    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if not 1 <= args.max_length <= 512:
        raise ValueError("--max-length must be between 1 and 512 for BERT.")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be greater than 0.")
    if args.weight_decay < 0:
        raise ValueError("--weight-decay must be non-negative.")
    if not 0 <= args.warmup_ratio <= 1:
        raise ValueError("--warmup-ratio must be between 0 and 1.")
    if args.seed < 0:
        raise ValueError("--seed must be non-negative.")
    if not args.output_dir.strip():
        raise ValueError("--output-dir must not be empty.")

    output_dir = resolve_path(args.output_dir)
    if os.path.exists(output_dir) and not os.path.isdir(output_dir):
        raise ValueError("--output-dir points to an existing file, not a directory.")


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


def compute_classification_metrics(
    true_labels: list[int],
    predictions: list[int],
) -> dict[str, float]:
    """Compute accuracy and macro F1 for multi-class classification."""
    return {
        "accuracy": float(accuracy_score(true_labels, predictions)),
        "macro_f1": float(
            f1_score(true_labels, predictions, average="macro", zero_division=0)
        ),
    }


@torch.no_grad()
def predict_split(
    model: BertForSequenceClassification,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[float, list[int], list[int], list[float]]:
    """Return average loss, true labels, predicted class ids, and confidence."""
    model.eval()
    total_loss = 0.0
    true_labels: list[int] = []
    predictions: list[int] = []
    confidences: list[float] = []

    for batch in dataloader:
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = model(**batch)
        probs = torch.softmax(outputs.logits, dim=-1)
        confidence, preds = torch.max(probs, dim=-1)

        total_loss += outputs.loss.item()
        true_labels.extend(batch["labels"].cpu().tolist())
        predictions.extend(preds.cpu().tolist())
        confidences.extend(confidence.cpu().tolist())

    avg_loss = total_loss / max(len(dataloader), 1)
    return avg_loss, true_labels, predictions, confidences


def build_prediction_frame(
    df: pd.DataFrame,
    true_labels: list[int],
    predictions: list[int],
    confidences: list[float],
) -> pd.DataFrame:
    """Build a prediction dataframe and verify row alignment."""
    result = df.reset_index(drop=True).copy()

    if len(result) != len(true_labels):
        raise ValueError("True label count does not match dataframe row count.")
    if len(result) != len(predictions):
        raise ValueError("Prediction count does not match dataframe row count.")
    if len(result) != len(confidences):
        raise ValueError("Confidence count does not match dataframe row count.")

    expected_true_labels = result["label_id"].tolist()
    if expected_true_labels != true_labels:
        raise ValueError(
            "Row alignment check failed: dataframe label_id values do not match "
            "the true labels observed during prediction."
        )

    result.insert(0, "row_id", result.index)
    result["pred_label_id"] = predictions
    result["pred_label"] = result["pred_label_id"].map(ID_TO_LABEL)
    result["pred_confidence"] = confidences

    if result["pred_label"].isna().any():
        raise ValueError("Predicted label ids could not be mapped back to label names.")

    return result


def compute_metrics_from_prediction_frame(prediction_df: pd.DataFrame) -> dict[str, float]:
    """Compute metrics directly from a saved-format prediction dataframe."""
    return compute_classification_metrics(
        true_labels=prediction_df["label_id"].tolist(),
        predictions=prediction_df["pred_label_id"].tolist(),
    )


def save_prediction_csv(
    prediction_df: pd.DataFrame,
    output_path: str,
) -> None:
    """Save a teammate-friendly prediction file."""
    prediction_df.to_csv(output_path, index=False)


def save_training_history(history: list[dict[str, float]], output_path: str) -> None:
    """Save per-epoch training history."""
    history_df = pd.DataFrame(history)
    history_df.to_csv(output_path, index=False)


def save_metrics_summary(
    metrics_rows: list[dict[str, object]],
    output_path: str,
) -> None:
    """Save final validation/test metrics for downstream analysis."""
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(output_path, index=False)


def verify_saved_prediction_metrics(
    prediction_path: str,
    expected_accuracy: float,
    expected_macro_f1: float,
    tolerance: float = 1e-12,
) -> dict[str, float]:
    """Recompute metrics from a saved prediction file and verify consistency."""
    prediction_df = pd.read_csv(prediction_path)
    recomputed_metrics = compute_metrics_from_prediction_frame(prediction_df)

    if abs(recomputed_metrics["accuracy"] - expected_accuracy) > tolerance:
        raise ValueError(
            f"Accuracy mismatch for '{prediction_path}': "
            f"expected {expected_accuracy:.12f}, "
            f"recomputed {recomputed_metrics['accuracy']:.12f}"
        )
    if abs(recomputed_metrics["macro_f1"] - expected_macro_f1) > tolerance:
        raise ValueError(
            f"Macro F1 mismatch for '{prediction_path}': "
            f"expected {expected_macro_f1:.12f}, "
            f"recomputed {recomputed_metrics['macro_f1']:.12f}"
        )

    return recomputed_metrics


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
    validate_args(args)
    set_seed(args.seed)

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
    print(f"[config] output_dir={output_dir}")

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
        val_loss, val_true_labels, val_predictions, _ = predict_split(
            model=model,
            dataloader=val_loader,
            device=device,
        )
        if val_true_labels != val_df["label_id"].tolist():
            raise ValueError(
                "Validation true labels do not match val_df['label_id']; "
                "row alignment may have been broken."
            )
        val_metrics = compute_classification_metrics(
            true_labels=val_true_labels,
            predictions=val_predictions,
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
            }
        )

        print(
            f"[epoch {epoch}/{args.epochs}] "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"val_accuracy={val_metrics['accuracy']:.4f}  "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            print(f"[checkpoint] Saved best model to '{checkpoint_dir}'")

    save_training_history(
        history=history,
        output_path=os.path.join(output_dir, "training_history.csv"),
    )

    best_model = BertForSequenceClassification.from_pretrained(checkpoint_dir)
    best_model.to(device)

    val_loss, val_true_labels, val_predictions, val_confidences = predict_split(
        model=best_model,
        dataloader=val_loader,
        device=device,
    )
    test_loss, test_true_labels, test_predictions, test_confidences = predict_split(
        model=best_model,
        dataloader=test_loader,
        device=device,
    )

    val_prediction_df = build_prediction_frame(
        df=val_df,
        true_labels=val_true_labels,
        predictions=val_predictions,
        confidences=val_confidences,
    )
    test_prediction_df = build_prediction_frame(
        df=test_df,
        true_labels=test_true_labels,
        predictions=test_predictions,
        confidences=test_confidences,
    )

    save_prediction_csv(
        prediction_df=val_prediction_df,
        output_path=os.path.join(output_dir, "val_predictions.csv"),
    )
    save_prediction_csv(
        prediction_df=test_prediction_df,
        output_path=os.path.join(output_dir, "test_predictions.csv"),
    )

    val_metrics = compute_metrics_from_prediction_frame(val_prediction_df)
    test_metrics = compute_metrics_from_prediction_frame(test_prediction_df)
    val_saved_metrics = verify_saved_prediction_metrics(
        prediction_path=os.path.join(output_dir, "val_predictions.csv"),
        expected_accuracy=val_metrics["accuracy"],
        expected_macro_f1=val_metrics["macro_f1"],
    )
    test_saved_metrics = verify_saved_prediction_metrics(
        prediction_path=os.path.join(output_dir, "test_predictions.csv"),
        expected_accuracy=test_metrics["accuracy"],
        expected_macro_f1=test_metrics["macro_f1"],
    )

    save_metrics_summary(
        metrics_rows=[
            {
                "split": "val",
                "rows": len(val_df),
                "loss": val_loss,
                "accuracy": val_saved_metrics["accuracy"],
                "macro_f1": val_saved_metrics["macro_f1"],
            },
            {
                "split": "test",
                "rows": len(test_df),
                "loss": test_loss,
                "accuracy": test_saved_metrics["accuracy"],
                "macro_f1": test_saved_metrics["macro_f1"],
            },
        ],
        output_path=os.path.join(output_dir, "metrics_summary.csv"),
    )

    print(f"[done] Best validation loss: {best_val_loss:.4f}")
    print(
        f"[done] Final validation loss: {val_loss:.4f}  "
        f"accuracy={val_metrics['accuracy']:.4f}  "
        f"macro_f1={val_metrics['macro_f1']:.4f}"
    )
    print(
        f"[done] Final test loss: {test_loss:.4f}  "
        f"accuracy={test_metrics['accuracy']:.4f}  "
        f"macro_f1={test_metrics['macro_f1']:.4f}"
    )
    print(f"[done] Wrote '{output_dir}/training_history.csv'")
    print(f"[done] Wrote '{output_dir}/val_predictions.csv'")
    print(f"[done] Wrote '{output_dir}/test_predictions.csv'")
    print(f"[done] Wrote '{output_dir}/metrics_summary.csv'")


if __name__ == "__main__":
    main()
