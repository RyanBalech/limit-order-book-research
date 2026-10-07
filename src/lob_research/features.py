import numpy as np

from .data import LOBSTERData

FEATURE_NAMES = (
    "spread",
    "mid_price",
    "queue_imbalance",
    "microprice_displacement",
    "depth_imbalance",
    "signed_event_size",
    "ofi",
)


def _safe_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    return np.divide(num, den, out=np.zeros_like(num, dtype=float), where=den != 0)


def build_features(data: LOBSTERData, *, depth: int = 5, price_scale: float = 1e-4) -> np.ndarray:
    """Build causal event-time microstructure features from aligned snapshots."""
    if depth < 1 or depth > data.levels:
        raise ValueError("depth must be between 1 and the available number of levels")

    book = data.book
    ask_p = book[:, 0::4] * price_scale
    ask_q = book[:, 1::4]
    bid_p = book[:, 2::4] * price_scale
    bid_q = book[:, 3::4]

    best_ask, best_bid = ask_p[:, 0], bid_p[:, 0]
    ask_size, bid_size = ask_q[:, 0], bid_q[:, 0]
    spread = best_ask - best_bid
    mid = 0.5 * (best_ask + best_bid)
    queue_imbalance = _safe_ratio(bid_size - ask_size, bid_size + ask_size)
    microprice = _safe_ratio(best_ask * bid_size + best_bid * ask_size, bid_size + ask_size)
    micro_disp = microprice - mid

    bid_depth = bid_q[:, :depth].sum(axis=1)
    ask_depth = ask_q[:, :depth].sum(axis=1)
    depth_imbalance = _safe_ratio(bid_depth - ask_depth, bid_depth + ask_depth)

    event_size = data.messages[:, 3]
    direction = data.messages[:, 5]
    # LOBSTER direction is the resting order side. For executions it is the
    # opposite of aggressor trade direction; this feature is NOT signed volume.
    signed_event_size = event_size * direction

    ofi = np.zeros(len(book), dtype=float)
    if len(book) > 1:
        bp, bq, ap, aq = best_bid, bid_size, best_ask, ask_size
        bid_flow = (bp[1:] >= bp[:-1]) * bq[1:] - (bp[1:] <= bp[:-1]) * bq[:-1]
        ask_flow = (ap[1:] <= ap[:-1]) * aq[1:] - (ap[1:] >= ap[:-1]) * aq[:-1]
        ofi[1:] = bid_flow - ask_flow

    return np.column_stack(
        [spread, mid, queue_imbalance, micro_disp, depth_imbalance, signed_event_size, ofi]
    )
