# Limit Order Book Research

A reproducible research project on **order-flow imbalance and short-horizon mid-price prediction** from limit order book data.

The project separates three questions: is there measurable short-horizon signal in order-book state and event flow; do nonlinear/deep models improve out-of-sample prediction over interpretable microstructure baselines; and do improvements survive chronological evaluation, uncertainty analysis, and simple trading-cost sanity checks?

## Research status

| Component | Status |
|---|---|
| LOBSTER message/order-book parser | Implemented |
| Microstructure features | Implemented |
| Forward mid-price labels | Implemented |
| Chronological split utilities | Implemented |
| Unit tests + CI | Implemented |
| Logistic regression baseline | Implemented |
| Gradient-boosted trees | Implemented |
| DeepLOB-style PyTorch model | Architecture + sequence windows implemented |
| Multi-day walk-forward study | Requires multi-day data |
| Cost-aware trading sanity check | Planned |

## Data

The initial reproducibility target is the public **LOBSTER sample format**. A paired message file and order-book file represent the same event sequence. Raw market data is deliberately not committed; place paired CSV files under `data/raw/`.

Public samples are useful for validating parsing, feature engineering, and the experimental pipeline. A single sample day is **not** treated as evidence of cross-day or cross-regime generalization.

## Initial features

- quoted spread and mid-price
- top-of-book queue imbalance
- microprice displacement from mid
- multi-level depth imbalance
- signed event size
- order-flow imbalance (OFI)

## Prediction target

At event index `t`, the target uses a strictly future mid-price at `t + h`. A predeclared relative threshold maps the future return to down, stationary, or up. Features at `t` never use future information.

## Evaluation principles

- chronological splits, never random market-time shuffling
- preprocessing fit on training data only
- class balance and naive baselines reported
- model selection separated from final test interval
- uncertainty across days/seeds where data permits
- predictive metrics separated from P&L claims
- spread/fees included before economic claims

See `docs/EXPERIMENT_PROTOCOL.md`.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

## Scope

This is a quantitative-research artifact, not investment advice or a production trading system. Results are added only after reproducible experiment runs.
