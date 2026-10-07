"""Render the published feature study without retraining or selecting a model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/published/feature_ablation.json")
    parser.add_argument("--output", default="docs/figures/feature_ablation.svg")
    args = parser.parse_args()
    groups = json.loads(Path(args.input).read_text())["groups"]
    names = list(groups)
    labels = ["price + spread", "book imbalance", "event flow", "without midpoint", "all features"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    axes[0].barh(labels, [groups[name]["test"]["macro_f1"] for name in names], color="steelblue")
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Test macro-F1")
    axes[0].set_xlim(0, .4)
    axes[0].set_title("Validation-selected feature arms")
    for name, label in [("all", "all"), ("all_without_midprice", "without midpoint")]:
        axes[1].plot(range(1, 7), [block["macro_f1"] for block in groups[name]["consecutive_test_blocks"]],
                     marker="o", label=label)
    axes[1].set_xlabel("Consecutive test interval")
    axes[1].set_ylabel("Macro-F1")
    axes[1].set_title("Within-day stability")
    axes[1].legend(frameon=False)
    counts = np.array(groups["all_without_midprice"]["test"]["confusion_true_rows_predicted_columns"])
    normalized = counts / counts.sum(1, keepdims=True)
    axes[2].imshow(normalized, vmin=0., vmax=1., cmap="Blues")
    for i in range(3):
        for j in range(3):
            axes[2].text(j, i, f"{100 * normalized[i, j]:.2f}%", ha="center", va="center",
                         color="white" if normalized[i, j] > .5 else "black")
    axes[2].set_xticks(range(3), ["down", "stationary", "up"])
    axes[2].set_yticks(range(3), ["down", "stationary", "up"])
    axes[2].set_xlabel("Predicted")
    axes[2].set_ylabel("True")
    axes[2].set_title("No-midpoint model: row-normalized")
    fig.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, metadata={"Date": None})


if __name__ == "__main__":
    main()
