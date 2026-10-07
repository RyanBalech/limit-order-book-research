from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lob_research.baselines import balanced_accuracy, macro_f1, majority_predict
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
    parser.add_argument("--output", default="results/gradient_boosted_baseline.json")
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
    except ImportError as exc:
        raise SystemExit('Install ML dependencies with: pip install -e ".[ml]"') from exc

    data = load_lobster_pair(args.messages, args.book)
    x_all = build_features(data, depth=args.depth)
    valid_idx, y = forward_midprice_labels(
        data.book, horizon=args.horizon, threshold=args.threshold
    )
    x = x_all[valid_idx]
    split = chronological_split(len(y), purge=args.horizon)
    x_train, x_val, x_test = x[split.train], x[split.validation], x[split.test]
    y_train, y_val, y_test = y[split.train], y[split.validation], y[split.test]

    candidates = [
        {"learning_rate": 0.05, "max_iter": 100, "max_leaf_nodes": 15, "l2_regularization": 0.0},
        {"learning_rate": 0.05, "max_iter": 200, "max_leaf_nodes": 31, "l2_regularization": 0.1},
        {"learning_rate": 0.10, "max_iter": 150, "max_leaf_nodes": 31, "l2_regularization": 1.0},
    ]
    trials = []
    best = None
    best_score = -1.0
    for params in candidates:
        model = HistGradientBoostingClassifier(random_state=args.seed, **params).fit(x_train, y_train)
        score = metrics(y_val, model.predict(x_val))
        trials.append({"params": params, "validation": score})
        if score["macro_f1"] > best_score:
            best_score = score["macro_f1"]
            best = params

    assert best is not None
    # Select hyperparameters on validation only, then refit on all pre-test observations.
    pretest = np.concatenate([split.train, split.validation])
    final_model = HistGradientBoostingClassifier(random_state=args.seed, **best).fit(x[pretest], y[pretest])
    test_metrics = metrics(y_test, final_model.predict(x_test))
    majority = metrics(y_test, majority_predict(y[pretest], len(y_test)))

    result = {
        "experiment": "gradient_boosted_microstructure_baseline",
        "feature_names": list(FEATURE_NAMES),
        "horizon_events": args.horizon,
        "threshold": args.threshold,
        "depth_levels": args.depth,
        "n_observations": len(y),
        "selection_metric": "validation_macro_f1",
        "candidate_trials": trials,
        "selected_params": best,
        "test": {"majority": majority, "hist_gradient_boosting": test_metrics},
        "warning": (
            "Hyperparameters are selected only on the chronological validation interval. "
            "A single sample file remains a pipeline study, not evidence of cross-day "
            "generalization or trading profitability."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
