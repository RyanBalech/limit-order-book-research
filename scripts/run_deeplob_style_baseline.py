from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from lob_research.baselines import Standardizer, balanced_accuracy, macro_f1, majority_predict
from lob_research.data import load_lobster_pair
from lob_research.deep_models import DeepLOBStyleClassifier, make_sequence_windows
from lob_research.features import FEATURE_NAMES, build_features
from lob_research.labels import forward_midprice_labels
from lob_research.splits import chronological_split


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {"macro_f1": macro_f1(y, pred), "balanced_accuracy": balanced_accuracy(y, pred)}


def encode(y: np.ndarray) -> np.ndarray:
    return (y + 1).astype(np.int64)


def decode(y: np.ndarray) -> np.ndarray:
    return (y - 1).astype(np.int8)


@torch.no_grad()
def predict(model: nn.Module, x: torch.Tensor, batch_size: int) -> np.ndarray:
    model.eval()
    loader = DataLoader(TensorDataset(x), batch_size=batch_size, shuffle=False)
    out = []
    for (xb,) in loader:
        out.append(model(xb).argmax(dim=1).cpu())
    return torch.cat(out).numpy()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("messages")
    p.add_argument("book")
    p.add_argument("--output", default="results/deeplob_style_baseline.json")
    p.add_argument("--horizon", type=int, default=100)
    p.add_argument("--threshold", type=float, default=0.0)
    p.add_argument("--depth", type=int, default=5)
    p.add_argument("--sequence-length", type=int, default=50)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data = load_lobster_pair(args.messages, args.book)
    x_all = build_features(data, depth=args.depth)
    valid_idx, y = forward_midprice_labels(data.book, horizon=args.horizon, threshold=args.threshold)
    x = x_all[valid_idx]
    split = chronological_split(len(y), purge=max(args.horizon, args.sequence_length))

    scaler = Standardizer().fit(x[split.train])
    x_scaled = scaler.transform(x).astype(np.float32)

    # Build windows separately inside each chronological block so no sequence crosses a split boundary.
    def block(indices: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        xb = torch.from_numpy(x_scaled[indices])
        yb = torch.from_numpy(encode(y[indices]))
        return make_sequence_windows(xb, yb, args.sequence_length)

    x_train, y_train = block(split.train)
    x_val, y_val = block(split.validation)
    x_test, y_test = block(split.test)

    counts = torch.bincount(y_train, minlength=3).float()
    weights = counts.sum() / (3.0 * counts.clamp_min(1.0))
    model = DeepLOBStyleClassifier(n_features=x.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=weights)
    train_loader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True
    )

    best_state = None
    best_val = -1.0
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))

        val_pred = decode(predict(model, x_val, args.batch_size))
        val_true = decode(y_val.numpy())
        val_metrics = metrics(val_true, val_pred)
        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "validation": val_metrics,
        })
        if val_metrics["macro_f1"] > best_val:
            best_val = val_metrics["macro_f1"]
            best_state = deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    test_true = decode(y_test.numpy())
    test_pred = decode(predict(model, x_test, args.batch_size))
    train_labels = decode(y_train.numpy())

    result = {
        "experiment": "deeplob_style_sequence_baseline",
        "feature_names": list(FEATURE_NAMES),
        "sequence_length": args.sequence_length,
        "horizon_events": args.horizon,
        "threshold": args.threshold,
        "selection_metric": "validation_macro_f1",
        "best_validation_macro_f1": best_val,
        "epochs_completed": len(history),
        "history": history,
        "test": {
            "majority": metrics(test_true, majority_predict(train_labels, len(test_true))),
            "deeplob_style": metrics(test_true, test_pred),
        },
        "warning": (
            "Compact DeepLOB-style CNN-LSTM over engineered features. A single sample day is "
            "not evidence of cross-day generalization or trading profitability."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
