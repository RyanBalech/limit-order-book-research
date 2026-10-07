import numpy as np


def mid_prices(book: np.ndarray, *, price_scale: float = 1e-4) -> np.ndarray:
    if book.ndim != 2 or book.shape[1] < 4:
        raise ValueError("book must contain at least one ask/bid level")
    return 0.5 * (book[:, 0] + book[:, 2]) * price_scale


def forward_midprice_labels(
    book: np.ndarray,
    *,
    horizon: int,
    threshold: float = 0.0,
    price_scale: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Return valid row indices and {-1,0,+1} labels using only future mid-price movement."""
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    mid = mid_prices(book, price_scale=price_scale)
    if horizon >= len(mid):
        return np.empty(0, dtype=int), np.empty(0, dtype=np.int8)
    current = mid[:-horizon]
    future = mid[horizon:]
    returns = np.divide(future - current, current, out=np.zeros_like(current), where=current != 0)
    labels = np.zeros(len(returns), dtype=np.int8)
    labels[returns > threshold] = 1
    labels[returns < -threshold] = -1
    return np.arange(len(returns)), labels
