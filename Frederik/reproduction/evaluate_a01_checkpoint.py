"""Evaluate CTNet's released subject-A01 checkpoint on BCI Competition IV-2a.

This script follows the preprocessing and test-time standardization in the
authors' repository.  It evaluates (rather than retrains) the released model,
which is the first of two useful reproducibility checks.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCAL_PACKAGES = ROOT / ".tools" / "repro"
if LOCAL_PACKAGES.exists():
    sys.path.insert(0, str(LOCAL_PACKAGES))
CURRENT_MNE = ROOT / ".tools" / "mne_current"
if CURRENT_MNE.exists():
    # MNE 1.5.1 (the repository pin) cannot read GDF under NumPy 2.  A current
    # MNE is preferred solely for file parsing; all analysis choices stay fixed.
    sys.path.insert(0, str(CURRENT_MNE))

import mne
import numpy as np
import scipy.io
import torch
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from torch import Tensor, nn


def numberClassChannel(database_type: str) -> tuple[int, int]:
    if database_type == "A":
        return 4, 22
    if database_type == "B":
        return 2, 3
    raise ValueError(f"Unknown database type: {database_type}")


# These class names and their structure intentionally match the released code.
# The checkpoint stores a complete pickled model under __main__, so compatible
# class definitions must exist when it is loaded.
class PatchEmbeddingCNN(nn.Module):
    def __init__(
        self,
        f1=8,
        kernel_size=64,
        D=2,
        pooling_size1=8,
        pooling_size2=8,
        dropout_rate=0.3,
        number_channel=22,
        emb_size=40,
    ):
        super().__init__()
        f2 = D * f1
        self.cnn_module = nn.Sequential(
            nn.Conv2d(1, f1, (1, kernel_size), (1, 1), padding="same", bias=False),
            nn.BatchNorm2d(f1),
            nn.Conv2d(
                f1,
                f2,
                (number_channel, 1),
                (1, 1),
                groups=f1,
                padding="valid",
                bias=False,
            ),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, pooling_size1)),
            nn.Dropout(dropout_rate),
            nn.Conv2d(f2, f2, (1, 16), padding="same", bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, pooling_size2)),
            nn.Dropout(dropout_rate),
        )
        self.projection = nn.Sequential(Rearrange("b e (h) (w) -> b (h w) e"))

    def forward(self, x: Tensor) -> Tensor:
        return self.projection(self.cnn_module(x))


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum("bhqd, bhkd -> bhqk", queries, keys)
        if mask is not None:
            energy.masked_fill_(~mask, torch.finfo(torch.float32).min)
        attention = F.softmax(energy / (self.emb_size**0.5), dim=-1)
        attention = self.att_drop(attention)
        output = torch.einsum("bhal, bhlv -> bhav", attention, values)
        output = rearrange(output, "b h n d -> b n (h d)")
        return self.projection(output)


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class ClassificationHead(nn.Sequential):
    def __init__(self, flatten_number, n_classes):
        super().__init__()
        self.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(flatten_number, n_classes))

    def forward(self, x):
        return self.fc(x)


class ResidualAdd(nn.Module):
    def __init__(self, fn, emb_size, drop_p):
        super().__init__()
        self.fn = fn
        self.drop = nn.Dropout(drop_p)
        self.layernorm = nn.LayerNorm(emb_size)

    def forward(self, x, **kwargs):
        return self.layernorm(self.drop(self.fn(x, **kwargs)) + x)


class TransformerEncoderBlock(nn.Sequential):
    def __init__(
        self,
        emb_size,
        num_heads=4,
        drop_p=0.5,
        forward_expansion=4,
        forward_drop_p=0.5,
    ):
        super().__init__(
            ResidualAdd(
                nn.Sequential(MultiHeadAttention(emb_size, num_heads, drop_p)),
                emb_size,
                drop_p,
            ),
            ResidualAdd(
                nn.Sequential(
                    FeedForwardBlock(
                        emb_size,
                        expansion=forward_expansion,
                        drop_p=forward_drop_p,
                    )
                ),
                emb_size,
                drop_p,
            ),
        )


class TransformerEncoder(nn.Sequential):
    def __init__(self, heads, depth, emb_size):
        super().__init__(
            *[TransformerEncoderBlock(emb_size, heads) for _ in range(depth)]
        )


class BranchEEGNetTransformer(nn.Sequential):
    def __init__(
        self,
        heads=2,
        depth=6,
        emb_size=16,
        number_channel=22,
        f1=20,
        kernel_size=64,
        D=2,
        pooling_size1=8,
        pooling_size2=8,
        dropout_rate=0.3,
        **kwargs,
    ):
        super().__init__(
            PatchEmbeddingCNN(
                f1=f1,
                kernel_size=kernel_size,
                D=D,
                pooling_size1=pooling_size1,
                pooling_size2=pooling_size2,
                dropout_rate=dropout_rate,
                number_channel=number_channel,
                emb_size=emb_size,
            )
        )


class PositioinalEncoding(nn.Module):  # spelling preserved for checkpoint compatibility
    def __init__(self, embedding, length=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.encoding = nn.Parameter(torch.randn(1, length, embedding))

    def forward(self, x):
        # The release hard-codes .cuda(); using x.device preserves the same
        # computation and also permits CPU evaluation.
        return self.dropout(x + self.encoding[:, : x.shape[1], :].to(x.device))


class EEGTransformer(nn.Module):
    def __init__(
        self,
        heads=2,
        emb_size=16,
        depth=6,
        database_type="A",
        eeg1_f1=8,
        eeg1_kernel_size=64,
        eeg1_D=2,
        eeg1_pooling_size1=8,
        eeg1_pooling_size2=8,
        eeg1_dropout_rate=0.3,
        eeg1_number_channel=22,
        flatten_eeg1=600,
        **kwargs,
    ):
        super().__init__()
        self.number_class, self.number_channel = numberClassChannel(database_type)
        self.emb_size = emb_size
        self.flatten_eeg1 = flatten_eeg1
        self.flatten = nn.Flatten()
        self.cnn = BranchEEGNetTransformer(
            heads,
            depth,
            emb_size,
            number_channel=self.number_channel,
            f1=eeg1_f1,
            kernel_size=eeg1_kernel_size,
            D=eeg1_D,
            pooling_size1=eeg1_pooling_size1,
            pooling_size2=eeg1_pooling_size2,
            dropout_rate=eeg1_dropout_rate,
        )
        self.position = PositioinalEncoding(emb_size, dropout=0.1)
        self.trans = TransformerEncoder(heads, depth, emb_size)
        self.classification = ClassificationHead(flatten_eeg1, self.number_class)

    def forward(self, x):
        cnn = self.cnn(x)
        cnn = self.position(cnn * math.sqrt(self.emb_size))
        features = cnn + self.trans(cnn)
        return features, self.classification(self.flatten(features))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    """Prefer a path relative to this reproducibility folder in saved output."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def torch_load_compat(path: Path, *, map_location, weights_only: bool):
    """Load under both PyTorch 1.13 and versions with weights_only support."""
    kwargs = {"map_location": map_location}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = weights_only
    return torch.load(path, **kwargs)


def read_epochs(gdf_path: Path, training: bool) -> np.ndarray:
    raw = mne.io.read_raw_gdf(gdf_path, verbose="WARNING")
    events, event_dict = mne.events_from_annotations(raw, verbose="WARNING")
    if training:
        event_id = {
            "Left": event_dict["769"],
            "Right": event_dict["770"],
            "Foot": event_dict["771"],
            "Tongue": event_dict["772"],
        }
    else:
        event_id = {"Unknown": event_dict["783"]}
    selected = events[np.isin(events[:, 2], list(event_id.values()))]
    raw.info["bads"] += ["EOG-left", "EOG-central", "EOG-right"]
    picks = mne.pick_types(
        raw.info, meg=False, eeg=True, eog=False, stim=False, exclude="bads"
    )
    epochs = mne.Epochs(
        raw,
        selected,
        event_id,
        picks=picks,
        tmin=0,
        tmax=3.996,
        preload=True,
        baseline=None,
        verbose="WARNING",
    )
    data = epochs.get_data()
    if data.shape != (288, 22, 1000):
        raise RuntimeError(f"Unexpected data shape for {gdf_path}: {data.shape}")
    return data


def preprocess(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cache = args.cache
    if cache.exists() and not args.rebuild_cache:
        loaded = np.load(cache)
        return loaded["train"], loaded["test"], loaded["test_labels"]

    train = read_epochs(args.train_gdf, training=True)
    test = read_epochs(args.test_gdf, training=False)
    train_labels = scipy.io.loadmat(args.train_labels)["classlabel"].reshape(-1)
    test_labels = scipy.io.loadmat(args.test_labels)["classlabel"].reshape(-1)
    if train_labels.shape != (288,) or test_labels.shape != (288,):
        raise RuntimeError(
            f"Unexpected label shapes: train={train_labels.shape}, test={test_labels.shape}"
        )
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        train=train,
        test=test,
        train_labels=train_labels,
        test_labels=test_labels,
    )
    return train, test, test_labels


def evaluate(args: argparse.Namespace) -> dict:
    train, test, labels_one_based = preprocess(args)
    train_mean = float(train.mean())
    train_std = float(train.std())
    standardized_test = (test - train_mean) / train_std
    labels = labels_one_based.astype(np.int64) - 1

    device = torch.device(args.device)
    # weights_only=False is required because the authors saved a complete model.
    # Only load the checkpoint obtained from the trusted CTNet repository.
    model = torch_load_compat(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )
    model = model.to(device).eval()

    predictions: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(standardized_test), args.batch_size):
            batch = torch.from_numpy(
                standardized_test[start : start + args.batch_size, None]
            ).float()
            _, logits = model(batch.to(device))
            predictions.append(logits.argmax(dim=1).cpu().numpy())
    predicted = np.concatenate(predictions)

    correct = int((predicted == labels).sum())
    accuracy = correct / len(labels)
    kappa = float(cohen_kappa_score(labels, predicted))
    matrix = confusion_matrix(labels, predicted, labels=[0, 1, 2, 3])
    provenance_paths = (
        args.cache,
        args.train_gdf,
        args.test_gdf,
        args.train_labels,
        args.test_labels,
        args.checkpoint,
    )
    result = {
        "experiment": "released-checkpoint evaluation",
        "subject": "A01",
        "dataset": "BCI Competition IV-2a",
        "n_test_trials": int(len(labels)),
        "correct": correct,
        "accuracy": accuracy,
        "accuracy_percent": 100 * accuracy,
        "cohen_kappa": kappa,
        "paper_table_2_accuracy_percent": 90.97,
        "absolute_difference_percentage_points": abs(100 * accuracy - 90.97),
        "confusion_matrix_rows_true_columns_predicted": matrix.tolist(),
        "train_mean": train_mean,
        "train_std": train_std,
        "device": str(device),
        "versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "mne": mne.__version__,
            "numpy": np.__version__,
        },
        "inputs": {
            portable_path(path): sha256(path)
            for path in provenance_paths
            if path.exists()
        },
        "inputs_not_present": [
            portable_path(path) for path in provenance_paths if not path.exists()
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with args.predictions.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["trial", "true_label", "predicted_label", "correct"])
        for index, (truth, pred) in enumerate(zip(labels, predicted), start=1):
            writer.writerow([index, int(truth + 1), int(pred + 1), int(truth == pred)])
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-gdf", type=Path, default=ROOT / "data/raw/A01T.gdf")
    parser.add_argument("--test-gdf", type=Path, default=ROOT / "data/raw/A01E.gdf")
    parser.add_argument(
        "--train-labels", type=Path, default=ROOT / "data/labels/A01T.mat"
    )
    parser.add_argument(
        "--test-labels", type=Path, default=ROOT / "data/labels/A01E.mat"
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=ROOT / "models/2a/model_1.pth"
    )
    parser.add_argument("--cache", type=Path, default=ROOT / "data/processed/A01.npz")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=72)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results/a01_checkpoint.json"
    )
    parser.add_argument(
        "--predictions", type=Path, default=ROOT / "results/a01_predictions.csv"
    )
    return parser.parse_args()


if __name__ == "__main__":
    outcome = evaluate(parse_args())
    print(json.dumps(outcome, indent=2))

