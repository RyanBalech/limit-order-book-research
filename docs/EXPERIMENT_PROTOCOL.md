# Experiment protocol

## Primary question

Do order-book state and order-flow features predict short-horizon mid-price direction out of sample, and do nonlinear/deep models add stable information beyond interpretable microstructure baselines?

## Data stages

### Stage A — public sample
Use a public LOBSTER-format sample to validate parsing, features, labels, model plumbing and reporting. Treat results as engineering/smoke evidence only.

### Stage B — multi-day study
Use multiple trading days. Split by time/day, not randomly by event. The final test period remains untouched until model and threshold choices are frozen.

## Baselines

1. majority/stationary predictor
2. logistic regression on interpretable features
3. gradient-boosted trees
4. DeepLOB-style neural model

A Transformer extension is secondary and retained only if it adds a scientifically useful comparison.

## Metrics

Primary metrics are macro F1 and balanced accuracy. Also report per-class precision/recall, confusion matrices, and probabilistic calibration when available. Economic diagnostics are separate.

## Leakage controls

- chronological splits only
- no future book states in features
- scalers fit on training data only
- model selection on validation only
- final test interval evaluated after choices are frozen
- purge overlapping forward-label horizons around split boundaries in the multi-day study

## Statistical reporting

For the multi-day study, report per-day results and paired uncertainty across the same test days. Avoid significance claims from a single public sample day.

## Reproducibility

Record ticker, date range, depth, event filtering, horizon, threshold, feature version, split boundaries, seed, package versions and model configuration with every result artifact.
