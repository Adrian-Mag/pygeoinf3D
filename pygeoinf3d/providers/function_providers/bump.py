"""Axis-separable C∞ compactly-supported bump function providers.

Each produced function is a product of 1-D bump functions, one per axis:

.. math::

    f_i(\\mathbf{x}) = C_i \\prod_{j=1}^{3}
        \\phi_k\\!\\left(\\frac{2(x_j - c_{ij})}{w_{ij}}\\right),
    \\qquad
    \\phi_k(t) =
    \\begin{cases}
        \\exp\\!\\left(\\dfrac{-k\\,t^2}{1 - t^2}\\right) & |t| < 1, \\\\
        0 & |t| \\ge 1.
    \\end{cases}

``widths`` :math:`w_{ij}` are **full widths**; the half-width per axis is
:math:`w_{ij}/2`.  The compact support of :math:`f_i` is the axis-aligned box
:math:`[c_{ij} - w_{ij}/2,\\, c_{ij} + w_{ij}/2]` for each coordinate *j*.

When ``normalized=True`` the normalisation constant :math:`C_i` is computed
numerically (via ``scipy.integrate.quad``) so that
:math:`\\int f_i\\,\\mathrm{d}V = 1`.
"""

from __future__ import annotations

import numpy as np
from scipy import integrate as scipy_integrate

from pygeoinf3d.providers.base import FunctionProvider3D
from pygeoinf3d.core.functions import Function3D


# ---------------------------------------------------------------------------
# 1-D bump helpers
# ---------------------------------------------------------------------------

def _bump1d_integral(k: float) -> float:
    """Compute :math:`\\int_{-1}^{1} \\phi_k(t)\\,\\mathrm{d}t`.

    Uses ``scipy.integrate.quad`` with a tight absolute tolerance.
    Results are cached after the first call for a given *k*.
    """
    # Local module-level cache — populated the first time each k is seen.
    cache = _bump1d_integral.__dict__.setdefault("_cache", {})
    if k not in cache:
        def phi(t):
            if abs(t) >= 1.0:
                return 0.0
            return float(np.exp(-k * t * t / (1.0 - t * t)))

        val, _ = scipy_integrate.quad(phi, -1.0, 1.0, limit=200, epsabs=1e-12)
        cache[k] = val
    return cache[k]


# ---------------------------------------------------------------------------
# BumpProvider3D
# ---------------------------------------------------------------------------

class BumpProvider3D(FunctionProvider3D):
    """Axis-separable :math:`C^\\infty` compactly-supported bump provider.

    Parameters
    ----------
    domain : Domain3D
        The computational domain on which the functions live.
    centers : array_like, shape ``(n, 3)`` or ``(3,)``
        Bump centres.  A shape-``(3,)`` input is treated as a single
        centre (``dim == 1``).
    widths : float or array_like
        Full widths of the compact support per axis.  Half-width is
        ``widths / 2``.

        * scalar → broadcast to ``(n, 3)``
        * shape ``(3,)`` → same per-axis for every kernel
        * shape ``(n, 3)`` → independent per-kernel, per-axis widths

        All values must be strictly positive.
    k : float, default ``1.0``
        Shape parameter for the 1-D profile :math:`\\phi_k`.  Larger
        values make the bump more concentrated near its centre.
        Must be positive.
    normalized : bool, default ``True``
        If ``True``, apply the numerically computed normalisation constant
        so that :math:`\\int f_i\\,\\mathrm{d}V = 1`.

    Raises
    ------
    ValueError
        If *centers* does not have shape ``(n, 3)``, if any width value is
        non-positive, if *k* ≤ 0, or if *widths* has an incompatible shape.

    Examples
    --------
    ::

        domain = BoxDomain(-2.0 * np.ones(3), 2.0 * np.ones(3))
        provider = BumpProvider3D(domain, np.array([[0.0, 0.0, 0.0]]),
                                  widths=1.0)
        f = provider.get_function(0)
    """

    def __init__(
        self,
        domain,
        centers,
        widths,
        *,
        k: float = 1.0,
        normalized: bool = True,
    ) -> None:
        centers = np.asarray(centers, dtype=float)
        if centers.ndim == 1:
            centers = centers[np.newaxis, :]
        if centers.ndim != 2 or centers.shape[1] != 3:
            raise ValueError(
                f"centers must have shape (n, 3) or (3,), got {centers.shape}"
            )
        n = len(centers)

        widths = np.asarray(widths, dtype=float)
        if widths.ndim == 0:
            widths = np.full((n, 3), float(widths))
        elif widths.ndim == 1 and widths.shape[0] == 3:
            widths = np.broadcast_to(widths[np.newaxis, :], (n, 3)).copy()
        elif widths.ndim == 2 and widths.shape == (n, 3):
            widths = widths.copy()
        else:
            raise ValueError(
                f"widths must be scalar, shape (3,), or shape ({n}, 3), "
                f"got {widths.shape}"
            )
        if np.any(widths <= 0):
            raise ValueError("All widths must be strictly positive.")
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")

        self._domain = domain
        self._centers = centers  # (n, 3)
        self._widths = widths    # (n, 3) — full widths
        self._k = float(k)
        self._normalized = normalized

    # ------------------------------------------------------------------
    # FunctionProvider3D interface
    # ------------------------------------------------------------------

    @property
    def dim(self) -> int:
        """Number of bump functions this provider can produce."""
        return len(self._centers)

    def get_function(self, i: int) -> Function3D:
        """Return the *i*-th bump function as a :class:`Function3D`.

        Parameters
        ----------
        i : int
            Index in ``range(dim)``.

        Returns
        -------
        Function3D
            Separable bump function with compact support
            :math:`[c_{ij} - w_{ij}/2,\\, c_{ij} + w_{ij}/2]` per axis.

        Raises
        ------
        IndexError
            If *i* is outside ``[0, dim)``.
        """
        if not (0 <= i < self.dim):
            raise IndexError(
                f"Index {i} is out of range [0, {self.dim})"
            )

        c = self._centers[i]   # (3,)
        w = self._widths[i]    # (3,) full widths
        hw = w * 0.5           # (3,) half-widths
        k = self._k

        if self._normalized:
            norm1d = _bump1d_integral(k)
            # Change of variable t = (x_j - c_j) / hw_j  →  dx_j = hw_j dt_j
            # ∫ phi_k(t) hw_j dt_j   summed over all axes  → product
            C = 1.0 / (norm1d ** 3 * hw[0] * hw[1] * hw[2])
        else:
            C = 1.0

        def _eval(x, _c=c, _hw=hw, _k=k, _C=C):
            x = np.asarray(x, dtype=float)
            if x.ndim == 1:
                t = (x - _c) / _hw          # (3,)
                if np.any(np.abs(t) >= 1.0):
                    return 0.0
                value = _C * np.prod(
                    np.exp(-_k * t ** 2 / (1.0 - t ** 2))
                )
                return float(value)
            # batch (n, 3)
            t = (x - _c) / _hw           # (b, 3) — broadcasts center (3,)
            inside = np.all(np.abs(t) < 1.0, axis=1)   # (b,)
            result = np.zeros(x.shape[0], dtype=float)
            if np.any(inside):
                ti = t[inside]             # (m, 3)
                vals = _C * np.prod(
                    np.exp(-_k * ti ** 2 / (1.0 - ti ** 2)), axis=1
                )
                result[inside] = vals
            return result

        return Function3D(
            self._domain,
            evaluate_callable=_eval,
            name=f"bump_{i}",
        )

    def evaluate_batch(self, i_start: int, i_end: int, points: np.ndarray) -> np.ndarray:
        """Vectorized evaluation of bumps ``i_start:i_end`` at *points*.

        Parameters
        ----------
        i_start, i_end : int
            Half-open row range ``[i_start, i_end)``.
        points : ndarray of shape ``(n_pts, 3)``

        Returns
        -------
        ndarray of shape ``(i_end - i_start, n_pts)``
        """
        points = np.asarray(points, dtype=float)                           # (n_pts, 3)
        c = self._centers[i_start:i_end]                                   # (B, 3)
        hw = self._widths[i_start:i_end] * 0.5                             # (B, 3)
        k = self._k
        # (B, n_pts, 3) scaled displacement
        t = (points[np.newaxis, :, :] - c[:, np.newaxis, :]) / hw[:, np.newaxis, :]
        inside = np.all(np.abs(t) < 1.0, axis=2)                          # (B, n_pts)
        # Build safe denominator: use 1.0 for outside points to avoid 0-denom
        safe_denom = np.where(inside[:, :, np.newaxis], 1.0 - t * t, 1.0)  # (B, n_pts, 3)
        exp_each = np.exp(-k * t * t / safe_denom)                         # (B, n_pts, 3)
        raw_product = np.prod(exp_each, axis=2)                            # (B, n_pts)
        result = np.where(inside, raw_product, 0.0)                        # (B, n_pts)
        if self._normalized:
            norm1d = _bump1d_integral(k)
            C = 1.0 / (norm1d ** 3 * hw[:, 0] * hw[:, 1] * hw[:, 2])     # (B,)
            result = C[:, np.newaxis] * result
        return result

    def __repr__(self) -> str:
        return (
            f"BumpProvider3D(dim={self.dim}, k={self._k}, "
            f"normalized={self._normalized})"
        )
