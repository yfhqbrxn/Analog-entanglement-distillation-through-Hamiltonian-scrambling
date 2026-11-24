"""
Utilities for finding the physical error rate `p` that satisfies
(1 - p/2)^(1 - r) * (2 - 3p/2)^r = 1 for a given repetition count `r`.

This module uses NumPy so the computations can be automatically vectorized
over many `r` values when needed.
"""

from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np

_LOG_TWO = float(np.log(2.0))
_DEFAULT_LOG_TOL = 1e-15


def _log_equation_value(p, r):
    """
    Return the logarithm of the left-hand side of
    (1 - p/2)^(1 - r) * (2 - 3p/2)^r = 1.

    Working in log-space keeps the evaluation numerically stable for large r.
    Arguments can be scalars or NumPy arrays (broadcasting is supported).
    """
    p_arr = np.asarray(p, dtype=np.float64)
    r_arr = np.asarray(r, dtype=np.float64)
    log_first = (1.0 - r_arr) * np.log1p(-0.5 * p_arr)
    log_second = r_arr * (_LOG_TWO + np.log1p(-0.75 * p_arr))
    return log_first + log_second


def compute_error_rate_root(r, tol: float = 1e-12, max_iter: int = 100):
    """
    Solve (1 - p/2)^(1 - r) * (2 - 3p/2)^r = 1 for p in [0, 2/3].

    Args:
        r: Scalar or array-like exponents; positive values ensure monotonicity.
        tol: Absolute tolerance for the bisection interval width.
        max_iter: Maximum number of bisection iterations.

    Returns:
        The unique root(s) p within [0, 2/3] matching the shape of `r`.

    Raises:
        ValueError: If any r == 0 (equation is satisfied for any p) or a root cannot be bracketed.
    """
    r_raw = np.asarray(r, dtype=np.float64)
    scalar_input = r_raw.ndim == 0
    orig_shape = r_raw.shape
    r_arr = np.atleast_1d(r_raw)

    if np.any(r_arr == 0):
        raise ValueError("r == 0 makes the equation identity; no unique root exists.")

    low = np.zeros_like(r_arr)
    high = np.full_like(r_arr, 2.0 / 3.0)
    f_low = _log_equation_value(low, r_arr)
    f_high = _log_equation_value(high, r_arr)

    same_sign = f_low * f_high > 0
    if np.any(same_sign):
        raise ValueError(
            "Failed to bracket root for r values at indices: {}".format(
                np.argwhere(same_sign).flatten().tolist()
            )
        )

    roots = np.empty_like(r_arr)
    converged = np.zeros_like(r_arr, dtype=bool)

    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        f_mid = _log_equation_value(mid, r_arr)

        interval_small = 0.5 * (high - low) < tol
        value_small = np.abs(f_mid) < _DEFAULT_LOG_TOL
        newly_converged = (~converged) & (interval_small | value_small)
        roots[newly_converged] = mid[newly_converged]
        converged |= newly_converged
        if np.all(converged):
            break

        go_low = (~converged) & (f_mid * f_low > 0)
        low[go_low], f_low[go_low] = mid[go_low], f_mid[go_low]

        go_high = (~converged) & (~go_low)
        high[go_high], f_high[go_high] = mid[go_high], f_mid[go_high]

    still_open = ~converged
    if np.any(still_open):
        roots[still_open] = 0.5 * (low[still_open] + high[still_open])

    if scalar_input:
        return float(np.asarray(roots).item())
    return roots.reshape(orig_shape)

# Sweep r values and evaluate the tolerable error rate curve.
r_list = np.linspace(0.2, 0.9999, 200)
t_vals = np.array([compute_error_rate_root(r=r) / 2 for r in r_list])

half_r = 0.5
half_tol = compute_error_rate_root(half_r) / 2

plt.style.use("seaborn-v0_8-darkgrid")
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(
    r_list,
    t_vals,
    color="#1f77b4",
    linewidth=2.2,
    marker="",
    label="tolerable error rate",
)
ax.fill_between(r_list, t_vals, color="#1f77b4", alpha=0.15)

max_idx = int(np.argmax(t_vals))
ax.scatter(r_list[max_idx], t_vals[max_idx], color="#d62728", zorder=5)
ax.annotate(
    f"peak ~ {t_vals[max_idx]:.3f}",
    xy=(r_list[max_idx], t_vals[max_idx]),
    xytext=(10, 15),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2),
    fontsize=12,
)

ax.axvline(half_r, color="#ff7f0e", linestyle=":", linewidth=1.4, label="m/n = 0.5")
ax.scatter(half_r, half_tol, color="#ff7f0e", zorder=6)
ax.annotate(
    "≈ 23%",
    xy=(half_r, half_tol),
    xytext=(24, -28),
    textcoords="offset points",
    ha="left",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
    arrowprops=dict(arrowstyle="->", color="#ff7f0e", lw=1.1),
    fontsize=13,
)

ax.set_xlabel("m/n ratio of measurement", fontsize=14)
ax.set_ylabel("tolerable error rate", fontsize=14)
ax.set_title("Tolerable Error Rate vs Measurement Ratio", fontsize=15)
ax.tick_params(axis="both", labelsize=12)
ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.6)
ax.legend(loc="upper left", fontsize=12)
ax.set_xlim(r_list[0], r_list[-1])
ax.set_ylim(0, 1.05 * t_vals.max())

plt.tight_layout()
# plt.show()
output_dir = Path(__file__).resolve().parents[1] / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(output_dir / "tolerable.svg", bbox_inches="tight", pad_inches=0.12)
# print(compute_error_rate_root(r=0.999999))
