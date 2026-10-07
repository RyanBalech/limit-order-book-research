from __future__ import annotations

import numpy as np


def threshold_positions(
    probabilities: np.ndarray,
    *,
    confidence: float,
    class_order: tuple[int, int, int] = (-1, 0, 1),
) -> np.ndarray:
    """Take a directional position only when model confidence clears a threshold."""
    if probabilities.ndim != 2 or probabilities.shape[1] != len(class_order):
        raise ValueError("probabilities must have one column per class")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must lie in [0, 1]")
    best = probabilities.argmax(axis=1)
    conf = probabilities.max(axis=1)
    positions = np.array([class_order[i] for i in best], dtype=np.int8)
    positions[conf < confidence] = 0
    return positions


def cost_aware_returns(
    positions: np.ndarray,
    future_mid_returns: np.ndarray,
    *,
    half_spread_bps: np.ndarray | float,
    fee_bps: float = 0.0,
) -> np.ndarray:
    """Simple one-period directional return after entry/exit spread and fees.

    This is a sanity check, not an execution simulator. A position pays two half-spreads
    plus fees whenever it is non-zero.
    """
    if positions.shape != future_mid_returns.shape:
        raise ValueError("positions and returns must have equal shape")
    gross = positions * future_mid_returns
    spread = np.asarray(half_spread_bps, dtype=float) * 2.0 / 10_000.0
    fees = 2.0 * fee_bps / 10_000.0
    active = positions != 0
    return gross - active * (spread + fees)


def trading_summary(net_returns: np.ndarray, positions: np.ndarray) -> dict[str, float]:
    active = positions != 0
    return {
        "mean_net_return": float(net_returns.mean()),
        "cumulative_net_return": float(net_returns.sum()),
        "trade_rate": float(active.mean()),
        "mean_net_return_when_active": float(net_returns[active].mean()) if active.any() else 0.0,
    }
