# AAPL public-sample baseline comparison

## Data and evaluation

The official LOBSTER AAPL level-10 sample covers 2012-06-21. We use the first five
book levels for features, a 100-event future midpoint target, and a zero return
threshold. There are 400,291 labeled observations: 200,827 down, 14,084 stationary,
and 185,380 up.

The chronological split has 240,174 training rows, 79,958 validation rows and 79,959
test rows, with 100-row gaps between intervals to avoid overlapping label horizons.
Both models select among three configurations using validation macro-F1, then refit
on the same train-plus-validation rows before one test evaluation. Logistic scaling
is refit only on those pre-test rows. Logistic regression uses balanced class weights;
the boosted model uses its default unweighted objective. This compares the stated
pipelines, not architecture alone.

## Measured results

| Method | Test macro-F1 | Test balanced accuracy |
|---|---:|---:|
| Majority class | 0.227316 | 0.333333 |
| Balanced logistic regression | 0.162659 | 0.349156 |
| Histogram gradient boosting | 0.317895 | 0.344253 |

Boosting improves macro-F1 by about 0.091 over the majority baseline. Logistic
regression has slightly higher balanced accuracy but lower macro-F1 than the
majority classifier. The metrics therefore do not support an unqualified claim that
one model is better on every criterion. Both balanced accuracies remain near 1/3.
Minority-class precision and temporal distribution shift deserve further analysis.

This is one sample day. Overlapping labels make rows dependent; no independent-row
confidence interval or cross-day significance claim is reported. There is no
demonstrated trading profit, and the deep model has not been evaluated in this table.

## Reproduce

```bash
pip install -e ".[dev,ml]"
python scripts/download_lobster_public_data.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/run_logistic_baseline.py data/raw/aapl_level10/AAPL_2012-06-21_34200000_57600000_message_10.csv data/raw/aapl_level10/AAPL_2012-06-21_34200000_57600000_orderbook_10.csv --output results/published/aapl_logistic_matched.json
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/run_gradient_boosted_baseline.py data/raw/aapl_level10/AAPL_2012-06-21_34200000_57600000_message_10.csv data/raw/aapl_level10/AAPL_2012-06-21_34200000_57600000_orderbook_10.csv --output results/published/aapl_boosted_matched.json
pytest -q
```

`results/published/environment.json` records package versions, thread limits and
SHA-256 hashes of both input files. The paired baseline workflow runs both scripts.
Version changes can alter histogram boosting slightly; use the recorded environment
for comparison. Original GitHub Actions output from
[run 37565178792](https://github.com/RyanBalech/limit-order-book-research/actions/runs/37565178792)
at commit `94a418071f39ccf7ec64a172d4b8b83e68002fc1` is preserved separately as
`aapl_level10_gradient_boosted.json` (macro-F1 0.320103).

Next steps: multiple independent days, walk-forward evaluation, per-class confusion
analysis, DeepLOB-style comparisons, and block/day-level uncertainty before any
economic interpretation.
