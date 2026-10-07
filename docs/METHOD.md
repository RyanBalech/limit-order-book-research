# Data contracts and research choices

## Observation time

LOBSTER message row t describes the event changing book row t−1 into book row t.
Features here use the **post-event** snapshot and the message at t. Prediction starts
after that event. Labels compare that midpoint to the midpoint at t+100; they do not
predict the event that was just observed.

The parser checks aligned row counts, finite values, non-decreasing timestamps and
non-negative quantities. Tests change future events/books and verify that past
features do not change. Purge tests check that an earlier interval's final forward
label ends before the next interval starts. Equal timestamps remain valid.

## Features

Let B/A be best bid/ask prices and qB/qA their displayed sizes.

- Queue imbalance: (qB−qA)/(qB+qA).
- Microprice: (A·qB+B·qA)/(qB+qA); feature is microprice minus midpoint.
- Depth imbalance: summed bid minus ask size, divided by their sum, across five levels.
- OFI: bid contribution minus ask contribution, accounting for quote-price moves and queue-size changes.
- `signed_event_size`: event size times **resting order side**, as encoded by LOBSTER.

The last feature is not aggressor-signed trade volume or net liquidity flow. Executing
a sell limit order corresponds to a buyer-initiated trade; cancellation signs need
separate interpretation. OFI uses book transitions instead. This distinction follows
the [official format](https://php.lobsterdata.com/info/DataStructure.php).

Absolute mid-price may act as a within-day time/regime proxy. The ablation removes it
while retaining price-relative spread and microprice displacement. The result is
consistent with poorer transfer of absolute price on this interval, but does not
prove that mechanism. It also exposes weak recovery of the stationary class.

## Model selection and dependence

Every feature arm selects among the same three boosting configurations on the same
validation interval. Early stopping is disabled: sklearn's default automatic early
stopping otherwise uses a random internal validation subset on a large training set.
The new ablation is therefore reported separately from the earlier baseline table.

The paired bootstrap resamples whole consecutive 1,000- or 5,000-event blocks using
identical block indices for both models. Confusion matrices are pooled before F1 is
computed; averaging block F1 would estimate a different quantity. A shorter final
block is retained. These intervals depend on approximate within-day block independence;
independent-day evaluation remains necessary.

No feature group is chosen by its test score to claim an untouched final benchmark.
All groups and per-class failures are reported. The CNN/LSTM and cost checks are
implemented extensions, with no claimed deep-model improvement or executed strategy.

## References

- [Cont, Kukanov & Stoikov, The Price Impact of Order Book Events](https://arxiv.org/abs/1011.6402): order-flow imbalance motivation.
- [Microsoft Qlib workflow configuration](https://github.com/microsoft/qlib/blob/main/examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml): explicit fitting and chronological evaluation segments.
- [LOBSTER data structure](https://php.lobsterdata.com/info/DataStructure.php): event timing, side convention and snapshot alignment.
