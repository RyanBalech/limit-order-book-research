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
    y_train, y_val, y_test = y[split.train], y[split.validation], y[split.test]

    candidates = [0.1, 1.0, 10.0]
    trials = []
    best_c, best_score = None, -1.0
    for c in candidates:
        model = LogisticRegression(
            C=c, max_iter=1000, class_weight="balanced", random_state=args.seed
        ).fit(x_train, y_train)
        score = metrics(y_val, model.predict(x_val))
        trials.append({"C": c, "validation": score})
        if score["macro_f1"] > best_score:
            best_c, best_score = c, score["macro_f1"]
    # Match the boosted baseline: select on validation, refit on train + validation.
    pretest = np.concatenate([split.train, split.validation])
    final_scaler = Standardizer().fit(x[pretest])
    final_model = LogisticRegression(
        C=best_c, max_iter=1000, class_weight="balanced", random_state=args.seed
    ).fit(final_scaler.transform(x[pretest]), y[pretest])
    test_metrics = metrics(y_test, final_model.predict(final_scaler.transform(x[split.test])))
    majority_test = majority_predict(y[pretest], len(y_test))

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
        "seed": args.seed,
        "selection_metric": "validation_macro_f1",
        "candidate_trials": trials,
        "selected_C": best_c,
        "refit": "train_plus_validation",
        "test": {
            "majority": metrics(y_test, majority_test),
            "logistic_regression": test_metrics,
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
