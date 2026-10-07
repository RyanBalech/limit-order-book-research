import numpy as np

from lob_research.costs import cost_aware_returns, threshold_positions, trading_summary


def test_threshold_positions_abstains_when_uncertain():
    p = np.array([[0.8, 0.1, 0.1], [0.34, 0.33, 0.33], [0.1, 0.1, 0.8]])
    assert threshold_positions(p, confidence=0.6).tolist() == [-1, 0, 1]


def test_costs_reduce_gross_directional_return():
    pos = np.array([1, -1, 0], dtype=np.int8)
    ret = np.array([0.001, -0.001, 0.0])
    net = cost_aware_returns(pos, ret, half_spread_bps=1.0, fee_bps=0.5)
    assert net[0] < 0.001
    assert net[1] < 0.001
    assert net[2] == 0.0
    assert trading_summary(net, pos)["trade_rate"] == 2 / 3
