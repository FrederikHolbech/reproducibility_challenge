"""Retrain CTNet for subject A01 using the released notebook protocol.

The default ``legacy`` validation mode intentionally preserves the indexing
used by the 2024 notebook so that a strict reproduction is possible.  Use
``--validation corrected`` for a non-overlapping 30% split within each batch.
Neither mode is a fixed hold-out split: the repository reshuffles the data at
every epoch.  This distinction is recorded in the result JSON.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from evaluate_a01_checkpoint import EEGTransformer, ROOT, torch_load_compat


def interaug(
    data: np.ndarray,
    labels_one_based: np.ndarray,
    *,
    n_aug: int,
    n_segments: int,
    batch_size: int,
    n_classes: int,
    n_channels: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    augmented_data = []
    augmented_labels = []
    records_per_class = n_aug * int(batch_size / n_classes)
    segment_width = 1000 // n_segments
    for class_id in range(1, n_classes + 1):
        class_data = data[labels_one_based == class_id]
        class_labels = labels_one_based[labels_one_based == class_id]
        generated = np.zeros(
            (records_per_class, 1, n_channels, 1000), dtype=np.float64
        )
        for record in range(records_per_class):
            source_trials = np.random.randint(0, class_data.shape[0], n_segments)
            for segment, source_trial in enumerate(source_trials):
                left = segment * segment_width
                right = (segment + 1) * segment_width
                generated[record, :, :, left:right] = class_data[
                    source_trial, :, :, left:right
                ]
        augmented_data.append(generated)
        augmented_labels.append(class_labels[:records_per_class])

    data_array = np.concatenate(augmented_data)
    label_array = np.concatenate(augmented_labels)
    order = np.random.permutation(len(data_array))
    data_tensor = torch.from_numpy(data_array[order]).float().to(device)
    label_tensor = torch.from_numpy(label_array[order] - 1).long().to(device)
    return data_tensor, label_tensor


def run(args: argparse.Namespace) -> dict:
    cached = np.load(args.cache)
    train_data = cached["train"]
    train_labels = cached["train_labels"].reshape(-1)
    test_data = cached["test"]
    test_labels = cached["test_labels"].reshape(-1).astype(np.int64) - 1
    if train_data.shape != (288, 22, 1000) or test_data.shape != (288, 22, 1000):
        raise RuntimeError("Expected A01 arrays with shape (288, 22, 1000)")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    if args.deterministic:
        torch.use_deterministic_algorithms(True)

    device = torch.device(args.device)
    model = EEGTransformer(
        heads=2,
        emb_size=16,
        depth=6,
        database_type="A",
        eeg1_f1=8,
        eeg1_kernel_size=64,
        eeg1_D=2,
        eeg1_pooling_size1=8,
        eeg1_pooling_size2=8,
        eeg1_dropout_rate=0.5,
        eeg1_number_channel=22,
        flatten_eeg1=240,
    ).to(device)

    # Match get_source_data(): one NumPy shuffle, then global train statistics.
    order = np.random.permutation(len(train_data))
    all_data = train_data[order, None]
    all_labels = train_labels[order]
    train_mean = float(all_data.mean())
    train_std = float(all_data.std())
    all_data = (all_data - train_mean) / train_std
    standardized_test = (test_data[:, None] - train_mean) / train_std

    dataset = TensorDataset(
        torch.from_numpy(all_data), torch.from_numpy(all_labels - 1)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.5, 0.999))
    criterion = nn.CrossEntropyLoss()
    best_loss = float("inf")
    best_epoch = -1
    history: list[dict] = []
    started = time.perf_counter()

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        model.train()
        validation_data = []
        validation_labels = []
        last_loss = None
        last_logits = None
        last_training_labels = None
        for batch_data, batch_labels in loader:
            validation_count = int(args.validation_ratio * len(batch_data))
            raw_train_data = batch_data[:-validation_count]
            raw_train_labels = batch_labels[:-validation_count]
            if args.validation == "legacy":
                # Exact 2024 notebook indexing: 70% validation with 41.7% of
                # each batch present in both training and validation.
                validation_data.append(batch_data[validation_count:])
                validation_labels.append(batch_labels[validation_count:])
            else:
                validation_data.append(batch_data[-validation_count:])
                validation_labels.append(batch_labels[-validation_count:])

            train_tensor = raw_train_data.float().to(device)
            label_tensor = raw_train_labels.long().to(device)
            aug_data, aug_labels = interaug(
                all_data,
                all_labels,
                n_aug=args.n_aug,
                n_segments=args.n_segments,
                batch_size=args.batch_size,
                n_classes=4,
                n_channels=22,
                device=device,
            )
            train_tensor = torch.cat((train_tensor, aug_data))
            label_tensor = torch.cat((label_tensor, aug_labels))
            _, logits = model(train_tensor)
            loss = criterion(logits, label_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            last_loss = loss
            last_logits = logits
            last_training_labels = label_tensor

        model.eval()
        val_data = torch.cat(validation_data).float()
        val_labels = torch.cat(validation_labels).long().to(device)
        val_outputs = []
        with torch.inference_mode():
            for start in range(0, len(val_data), args.batch_size):
                _, output = model(val_data[start : start + args.batch_size].to(device))
                val_outputs.append(output)
        val_logits = torch.cat(val_outputs)
        val_loss = criterion(val_logits, val_labels)
        val_accuracy = float((val_logits.argmax(1) == val_labels).float().mean())
        train_accuracy = float(
            (last_logits.argmax(1) == last_training_labels).float().mean()
        )
        record = {
            "epoch": epoch,
            "train_loss_last_batch": float(last_loss.detach().cpu()),
            "train_accuracy_last_batch": train_accuracy,
            "validation_loss": float(val_loss.detach().cpu()),
            "validation_accuracy": val_accuracy,
        }
        history.append(record)
        if record["validation_loss"] < best_loss:
            best_loss = record["validation_loss"]
            best_epoch = epoch
            torch.save(model.state_dict(), args.checkpoint)
            print(
                f"epoch={epoch:04d} train_acc={train_accuracy:.4f} "
                f"val_acc={val_accuracy:.4f} val_loss={best_loss:.7f}",
                flush=True,
            )

    model.load_state_dict(
        torch_load_compat(args.checkpoint, map_location=device, weights_only=True)
    )
    model.eval()
    test_outputs = []
    with torch.inference_mode():
        test_tensor = torch.from_numpy(standardized_test).float()
        for start in range(0, len(test_tensor), args.batch_size):
            _, output = model(test_tensor[start : start + args.batch_size].to(device))
            test_outputs.append(output.cpu())
    predictions = torch.cat(test_outputs).argmax(1).numpy()
    correct = int((predictions == test_labels).sum())
    accuracy = correct / len(test_labels)

    result = {
        "experiment": "training from scratch",
        "subject": "A01",
        "seed": args.seed,
        "validation_implementation": args.validation,
        "validation_note": (
            "2024 notebook overlap and per-epoch reshuffle preserved"
            if args.validation == "legacy"
            else "overlap fixed, but per-epoch reshuffle preserved"
        ),
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "elapsed_seconds": time.perf_counter() - started,
        "device": str(device),
        "correct": correct,
        "n_test_trials": int(len(test_labels)),
        "accuracy": accuracy,
        "accuracy_percent": 100 * accuracy,
        "cohen_kappa": float(cohen_kappa_score(test_labels, predictions)),
        "paper_table_2_accuracy_percent": 90.97,
        "absolute_difference_percentage_points": abs(100 * accuracy - 90.97),
        "confusion_matrix_rows_true_columns_predicted": confusion_matrix(
            test_labels, predictions, labels=[0, 1, 2, 3]
        ).tolist(),
        "versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with args.history.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    print(json.dumps(result, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=ROOT / "data/processed/A01.npz")
    parser.add_argument("--seed", type=int, default=966)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=72)
    parser.add_argument("--n-aug", type=int, default=3)
    parser.add_argument("--n-segments", type=int, default=8)
    parser.add_argument("--validation-ratio", type=float, default=0.3)
    parser.add_argument("--validation", choices=("legacy", "corrected"), default="legacy")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--checkpoint", type=Path, default=ROOT / "results/a01_retrained_state.pt"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results/a01_retrained.json"
    )
    parser.add_argument(
        "--history", type=Path, default=ROOT / "results/a01_training_history.csv"
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

