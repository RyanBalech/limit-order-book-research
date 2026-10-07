# Limit Order Book Research

Short-horizon mid-price direction prediction from LOBSTER messages and order-book
snapshots: causal microstructure features, purged chronological evaluation, and
comparisons between linear and tree-based models.

## Measured experiment

The public AAPL level-10 sample for 2012-06-21 yields **400,291 labeled events**.
Features use five book levels; labels use the midpoint 100 events ahead. Train,
validation and test are chronological with 100-event gaps. Hyperparameters are
selected on validation macro-F1, then models are refit on pre-test observations.

The original matched baselines achieved test macro-F1 0.163 (balanced logistic
regression) and 0.318 (histogram boosting), versus 0.227 for the majority classifier.

A follow-up feature ablation uses the same outer split and disables internal
random validation/early stopping in every boosting arm:

| Features | Test macro-F1 |
|---|---:|
| Price + spread | 0.3044 |
| Spread + book imbalances | 0.3451 |
| Spread + event flow | 0.3438 |
| All except absolute mid-price | **0.3537** |
| All features | 0.3079 |

Removing absolute price improves macro-F1 by 0.0459. Paired temporal-block bootstrap
intervals support that difference within this sample day. Per-class diagnostics
also reveal that stationary events are almost never recovered.

[Results, confusion matrices and uncertainty](docs/RESULTS.md) ·
[Raw ablation artifact](results/published/feature_ablation.json) ·
[Feature definitions and design choices](docs/METHOD.md)

![Feature groups, temporal stability and class-level failures](docs/figures/feature_ablation.svg)

## Reproduce

Python 3.10+. Raw market data is downloaded locally and is not committed.

```bash
python -m pip install -e ".[dev,ml]"
python scripts/download_lobster_public_data.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/run_feature_ablation.py \
  data/raw/aapl_level10/AAPL_2012-06-21_34200000_57600000_message_10.csv \
  data/raw/aapl_level10/AAPL_2012-06-21_34200000_57600000_orderbook_10.csv
python -m pytest -q
```

Artifacts include input SHA-256 hashes, split boundaries, validation trials,
selected parameters, per-class scores, six consecutive test intervals, and paired
block-bootstrap comparisons at two block sizes.

For the measured local stack, see [requirements-reproduce.txt](requirements-reproduce.txt)
(Python 3.12; critical package pins).

## Code

- [features.py](src/lob_research/features.py): queue/depth imbalance, microprice displacement and OFI.
- [labels.py](src/lob_research/labels.py), [splits.py](src/lob_research/splits.py): forward labels and horizon purging.
- [diagnostics.py](src/lob_research/diagnostics.py): confusion counts and paired temporal-block uncertainty.
- [deep_models.py](src/lob_research/deep_models.py): an additional CNN/LSTM baseline; not evaluated in the tables above.
- [costs.py](src/lob_research/costs.py): spread/fee sanity checks, separate from predictive scoring.

One public sample day cannot establish cross-day generalization or trading profits.
The follow-up is exploratory because the original test results were already observed.
