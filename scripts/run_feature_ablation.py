"""Exploratory feature ablation on a single purged chronological split."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier

from lob_research.data import load_lobster_pair
from lob_research.diagnostics import classification_diagnostics, paired_block_bootstrap_f1
from lob_research.features import FEATURE_NAMES, build_features
from lob_research.labels import forward_midprice_labels
from lob_research.splits import chronological_split

GROUPS = {"price_spread": [0, 1], "book_imbalance": [0, 2, 3, 4],
          "event_flow": [0, 5, 6], "all_without_midprice": [0, 2, 3, 4, 5, 6],
          "all": list(range(7))}
CANDIDATES = [
    {"learning_rate": .05, "max_iter": 100, "max_leaf_nodes": 15, "l2_regularization": 0.},
    {"learning_rate": .05, "max_iter": 200, "max_leaf_nodes": 31, "l2_regularization": .1},
    {"learning_rate": .10, "max_iter": 150, "max_leaf_nodes": 31, "l2_regularization": 1.},
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("messages")
    parser.add_argument("book")
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--output", default="results/feature_ablation.json")
    args = parser.parse_args()
    data = load_lobster_pair(args.messages, args.book)
    valid, y = forward_midprice_labels(data.book, horizon=args.horizon, threshold=0.)
    x = build_features(data, depth=5)[valid]
    split = chronological_split(len(y), purge=args.horizon)
    pretest = np.concatenate([split.train, split.validation])
    runs, predictions = {}, {}
    for name, columns in GROUPS.items():
        trials = []
        for params in CANDIDATES:
            model = HistGradientBoostingClassifier(random_state=2026, early_stopping=False,
                                                   **params).fit(x[split.train][:, columns], y[split.train])
            score = classification_diagnostics(y[split.validation],
                                               model.predict(x[split.validation][:, columns]))["macro_f1"]
            trials.append({"params": params, "validation_macro_f1": score})
        chosen = max(trials, key=lambda trial: trial["validation_macro_f1"])
        model = HistGradientBoostingClassifier(random_state=2026, early_stopping=False,
                                               **chosen["params"]).fit(x[pretest][:, columns], y[pretest])
        pred = model.predict(x[split.test][:, columns])
        predictions[name] = pred
        time_blocks = []
        for interval in np.array_split(np.arange(len(pred)), 6):
            time_blocks.append({"first_test_offset": int(interval[0]),
                                "last_test_offset": int(interval[-1]),
                                **classification_diagnostics(y[split.test][interval], pred[interval])})
        runs[name] = {"features": [FEATURE_NAMES[i] for i in columns],
                      "trials": trials, "selected_params": chosen["params"],
                      "test": classification_diagnostics(y[split.test], pred),
                      "consecutive_test_blocks": time_blocks}
        print(f"{name}: test macro-F1 {runs[name]['test']['macro_f1']:.6f}", flush=True)
    paired = {str(size): paired_block_bootstrap_f1(y[split.test], predictions["all"],
                                                  predictions["all_without_midprice"], block_size=size)
              for size in [1000, 5000]}
    result = {"experiment": "single_day_feature_ablation", "horizon_events": args.horizon,
              "threshold": 0., "depth": 5, "seed": 2026, "n_observations": len(y),
              "early_stopping": False, "selection": "validation macro-F1; refit train+validation",
              "split": {k: [int(v[0]), int(v[-1])] for k, v in
                        [("train", split.train), ("validation", split.validation), ("test", split.test)]},
              "inputs": {Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                         for p in [args.messages, args.book]},
              "environment": {"python": platform.python_version(), "numpy": np.__version__,
                              "sklearn": sklearn.__version__},
              "groups": runs, "all_minus_without_midprice_bootstrap": paired,
              "limitations": "Exploratory follow-up after the original test results. One day only; "
                              "block intervals assume approximate within-day dependence. "
                              "No group is selected using test metrics; no profitability claim."}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
