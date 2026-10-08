"""Turning log-space (multiplicative) attributions into dollars.

For a model fitted to log(price), SHAP values come out in log units:
``log ŷ = b + Σ_j a_j``. The grid needs dollars, and the dollar pieces have to
add up to the dollar prediction. Naive conversions fail at this. Taking
``exp(a_j)`` gives multipliers, not dollar amounts. Prorating
``a_j / Σa · (ŷ - e^b)`` breaks when the log terms cancel each other out.

Here the dollar contributions are defined as the **Shapley values of the
dollar game** ``v(S) = g(b + Σ_{j∈S} a_j)``, where ``g`` is the inverse of the
target transform (``exp`` for log). Each feature gets its average marginal
dollar effect over every order in which features could be "switched on".
This is the Shorrocks-Shapley decomposition that economists use to split
changes in a product into additive parts. It is efficient
(``e^b + Σ_j φ_j = ŷ`` exactly), symmetric, and unique.

For ``g = exp`` there is a closed form. Owen's multilinear extension
rewrites the Shapley weighted sum over coalitions as an integral::

    φ_j = e^b (m_j - 1) ∫₀¹ Π_{i≠j} (1 - t + t·m_i) dt,    m_i = e^{a_i}

The integrand is a polynomial of degree n-1 in t, so Gauss-Legendre
quadrature with ⌈n/2⌉ nodes computes it *exactly*. The cost is O(n²) per row,
vectorized over rows, with no sampling and no 2ⁿ enumeration.

The reference point is the log-space baseline ``b``, so ``base_value`` is
``exp(b)``, the value of the *geometric-mean* parcel rather than the
arithmetic mean prediction. In the literature this is baseline Shapley
(Sundararajan & Najmi, 2020) applied to the dollarized game. The grid uses
only how each contribution varies with its feature, never the base itself,
so this choice doesn't change any adjustment.

Other inverse transforms fall back to permutation sampling. Each sampled
permutation is exactly efficient, so the dollar contributions still add up
to the prediction exactly; only the split between features carries sampling
noise.
"""
from __future__ import annotations

from typing import Callable, Optional, Union

import numpy as np

Target = Union[str, Callable[[np.ndarray], np.ndarray], None]


def inverse_for(target: Target) -> Optional[Callable[[np.ndarray], np.ndarray]]:
    if target in (None, "identity", "linear"):
        return None
    if target == "log":
        return np.exp
    if target == "log1p":
        return np.expm1
    if target == "log10":
        return lambda z: np.power(10.0, z)
    if callable(target):
        return target
    raise ValueError(f"unknown target transform {target!r}; use 'identity', 'log', 'log1p', 'log10' or a callable inverse")


def to_dollars(base: np.ndarray, phi: np.ndarray, target: Target, n_permutations: int = 256,
               seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Map link-space ``(base, phi)`` to dollar-space ``(base$, phi$)``.

    ``base`` has shape (n_rows,) and ``phi`` has shape (n_rows, n_features).
    Returns arrays of the same shapes with ``base$ + phi$.sum(1) == g(base + phi.sum(1))``.
    """
    g = inverse_for(target)
    base = np.asarray(base, dtype=float)
    phi = np.asarray(phi, dtype=float)
    if g is None:
        return base, phi
    if target in ("log", "log1p", "log10"):
        scale = np.log(10.0) if target == "log10" else 1.0
        a = phi * scale
        b = base * scale
        dollars = _exp_shapley(b, a)
        base_d = np.exp(b) - (1.0 if target == "log1p" else 0.0)
        return base_d, dollars
    return g(base), _permutation_shapley(base, phi, g, n_permutations, seed)


def _exp_shapley(b: np.ndarray, a: np.ndarray) -> np.ndarray:
    n = a.shape[1]
    if n == 0:
        return a.copy()
    q = max(1, (n + 1) // 2)
    nodes, weights = np.polynomial.legendre.leggauss(q)
    t = (nodes + 1) / 2.0           # map [-1, 1] -> [0, 1]
    w = weights / 2.0
    m = np.exp(a)                                     # (rows, n)
    # factors[r, q, i] = 1 - t_q + t_q * m_ri  (always > 0)
    factors = 1.0 - t[None, :, None] + t[None, :, None] * m[:, None, :]
    log_f = np.log(factors)
    total = log_f.sum(axis=2, keepdims=True)          # log Π_i
    loo = np.exp(total - log_f)                       # Π_{i≠j}, via logs for stability
    integral = (loo * w[None, :, None]).sum(axis=1)   # (rows, n)
    return np.exp(b)[:, None] * (m - 1.0) * integral


def _permutation_shapley(base: np.ndarray, phi: np.ndarray, g: Callable, n_perm: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows, n = phi.shape
    out = np.zeros_like(phi)
    for _ in range(n_perm):
        order = rng.permutation(n)
        z = base.copy()
        prev = g(z)
        for j in order:
            z = z + phi[:, j]
            cur = g(z)
            out[:, j] += cur - prev
            prev = cur
    return out / n_perm
