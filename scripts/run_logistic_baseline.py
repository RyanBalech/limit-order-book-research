from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lob_research.baselines import Standardizer, balanced_accuracy, macro_f1, majority_predict
from lob_research.data import load_lobster_pair
from lob_research.features import FEATURE_NAMES, build_features
from lob_research.labels import forward_midprice_labels
from lob_research.splits import chronological_split


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {"macro_f1": macro_f1(y, pred), "balanced_accuracy": balanced_accuracy(y, pred)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("messages")
    parser.add_argument("book")
    parser.add_argument("--output", default="results/logistic_baseline.json")
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:
        raise SystemExit('Install ML dependencies with: pip install -e ".[ml]"') from exc

    data = load_lobster_pair(args.messages, args.book)
    x_all = build_features(data, depth=args.depth)
    valid_idx, y = forward_midprice_labels(
        data.book, horizon=args.horizon, threshold=args.threshold
    )
    x = x_all[valid_idx]

    split = chronological_split(len(y), purge=args.horizon)
    scaler = Standardizer().fit(x[split.train])
    x_train = scaler.transform(x[split.train])
    x_val = scaler.transform(x[split.validation])
    x_test = scaler.transform(x[split.test])
    y_train, y_val, y_test = y[split.train], y[split.validation], y[split.test]

    majority_val = majority_predict(y_train, len(y_val))
    majority_test = majority_predict(y_train, len(y_test))
    model = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=args.seed
    ).fit(x_train, y_train)

    result = {
        "experiment": "interpretable_logistic_baseline",
        "feature_names": list(FEATURE_NAMES),
        "horizon_events": args.horizon,
        "threshold": args.threshold,
        "depth_levels": args.depth,
        "n_observations": len(y),
        "class_counts": {
            str(int(cls)): int(np.sum(y == cls)) for cls in (-1, 0, 1)
        },
        "split_sizes": {
            "train": len(split.train),
            "validation": len(split.validation),
            "test": len(split.test),
            "purge": args.horizon,
        },
        "majority": {
            "validation": metrics(y_val, majority_val),
            "test": metrics(y_test, majority_test),
        },
        "logistic_regression": {
            "validation": metrics(y_val, model.predict(x_val)),
            "test": metrics(y_test, model.predict(x_test)),
        },
        "warning": (
            "A single sample file is a pipeline/smoke study, not evidence of "
            "cross-day generalization or trading profitability."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
