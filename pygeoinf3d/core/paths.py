"""Parameterized curves (paths) in 3D domains.

This module provides :class:`Path3D`: a representation of a parameterized
curve $\\gamma : [0, 1] \\to \\mathbb{R}^3$ living inside a
:class:`~pygeoinf3d.core.domain_base.Domain3D`.

Three construction modes are supported:

callable
    User supplies a raw parameterization $\\gamma(t)$ on $[t_\\text{start},
    t_\\text{end}]$.  Arc-length is approximated numerically.

polyline
    User supplies ordered waypoints; the path linearly interpolates between
    consecutive waypoints, parameterized by normalized chord-length.

point-sample
    Alias for *polyline* in Phase 2 (piecewise-linear interpolation).
    Smoother spline interpolation is deferred.

All modes expose a **unified callable interface**: ``path(t)`` where
$t \\in [0, 1]$ is the normalized arc-length parameter.  Scalar *t* returns
shape ``(3,)``; array *t* of shape ``(n,)`` returns shape ``(n, 3)``.
"""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _chord_lengths(waypoints: np.ndarray) -> np.ndarray:
    """Return the chord length of each segment in *waypoints*.

    Parameters
    ----------
    waypoints:
        Array of shape ``(n, 3)``, ``n >= 2``.

    Returns
    -------
    np.ndarray
        Shape ``(n-1,)`` — Euclidean distance between consecutive points.
    """
    diffs = np.diff(waypoints, axis=0)          # (n-1, 3)
    return np.sqrt(np.einsum("ij,ij->i", diffs, diffs))


def _build_polyline_evaluator(
    waypoints: np.ndarray,
) -> Tuple[Callable, float]:
    """Build a normalized-t evaluator for a piecewise-linear path.

    Parameters
    ----------
    waypoints:
        Array of shape ``(n, 3)``.

    Returns
    -------
    evaluator : Callable
        ``evaluator(t)`` where ``t`` is a scalar or ``(m,)`` array in
        ``[0, 1]``;  returns ``(3,)`` or ``(m, 3)`` respectively.
    arclength : float
        Total chord-length arc length.
    """
    segs = _chord_lengths(waypoints)           # (n-1,)
    total = float(segs.sum())
    if total == 0.0:
        raise ValueError("Degenerate path: all waypoints are identical.")

    # Cumulative arc-length fraction at each waypoint (including 0 at start).
    cum = np.concatenate([[0.0], np.cumsum(segs) / total])  # (n,)

    def _eval(t: np.ndarray | float):
        t_arr = np.asarray(t, dtype=float)
        scalar = t_arr.ndim == 0
        t_arr = np.atleast_1d(t_arr)
        t_arr = np.clip(t_arr, 0.0, 1.0)

        # For each query find the segment index using searchsorted.
        # np.searchsorted(cum, t_arr, side='right') - 1 gives the left
        # waypoint index; clamp to [0, n-2].
        idx = np.searchsorted(cum, t_arr, side="right") - 1
        idx = np.clip(idx, 0, len(segs) - 1)

        # Local parameter within the segment (handle zero-length segments).
        seg_len = segs[idx]                        # (m,)
        local_t = np.where(
            seg_len > 0.0,
            (t_arr - cum[idx]) / (cum[idx + 1] - cum[idx] + 1e-300),
            0.0,
        )
        local_t = np.clip(local_t, 0.0, 1.0)

        # Interpolate: shape (m, 3)
        p0 = waypoints[idx]                        # (m, 3)
        p1 = waypoints[idx + 1]                    # (m, 3)
        pts = p0 + local_t[:, None] * (p1 - p0)

        return pts[0] if scalar else pts

    return _eval, total


def _approx_arclength_callable(
    raw_gamma: Callable,
    t_start: float,
    t_end: float,
    n_pts: int,
) -> float:
    """Approximate total arc length of a smooth curve via a fine uniform grid.

    Parameters
    ----------
    raw_gamma:
        Callable ``gamma(t)`` accepting scalar *t* → ``(3,)`` or array
        ``(n,)`` → ``(n, 3)``.
    t_start, t_end:
        Parameter domain.
    n_pts:
        Number of sample points for the approximation.

    Returns
    -------
    float
        Approximate arc length.
    """
    ts = np.linspace(t_start, t_end, n_pts)
    pts = raw_gamma(ts)                # (n_pts, 3)
    diffs = np.diff(pts, axis=0)       # (n_pts-1, 3)
    segs = np.sqrt(np.einsum("ij,ij->i", diffs, diffs))
    return float(segs.sum())


def _build_callable_evaluator(
    raw_gamma: Callable,
    t_start: float,
    t_end: float,
    arclength: float,
    n_approx_pts: int,
) -> Callable:
    """Return a normalized-t evaluator for a smooth callable path.

    Internally builds a piecewise-linear skeleton on *n_approx_pts* points
    sampled uniformly in the raw parameter domain, then uses
    :func:`_build_polyline_evaluator` on that skeleton to map
    $t \\in [0,1]$ back to the original parameterization domain, before
    delegating to `raw_gamma`.

    Parameters
    ----------
    raw_gamma:
        Raw parameterization callable.
    t_start, t_end:
        Raw parameter domain.
    arclength:
        Pre-computed approximate total arc length.
    n_approx_pts:
        Skeleton resolution.

    Returns
    -------
    Callable
        ``evaluator(t)`` for normalized ``t ∈ [0, 1]``.
    """
    # Build a skeleton of (n_approx_pts, 3) points.
    ts_raw = np.linspace(t_start, t_end, n_approx_pts)
    skeleton = raw_gamma(ts_raw)       # (n_approx_pts, 3)

    # Build a chord-length normalized evaluator on the skeleton.
    # We use this only to map t_norm → raw_t, then delegate to raw_gamma.
    segs = _chord_lengths(skeleton)             # (n_approx_pts - 1,)
    total = float(segs.sum())
    if total == 0.0:
        raise ValueError(
            "Degenerate callable path: all sampled points are identical."
        )
    cum_norm = np.concatenate([[0.0], np.cumsum(segs) / total])

    def _eval(t: np.ndarray | float):
        t_arr = np.asarray(t, dtype=float)
        scalar = t_arr.ndim == 0
        t_arr = np.atleast_1d(t_arr)
        t_arr = np.clip(t_arr, 0.0, 1.0)

        # Map t_norm → raw_t by interpolating ts_raw via the chord skeleton.
        # Linear interpolation from normalized t → raw parameter.
        raw_t = np.interp(t_arr, cum_norm, ts_raw)

        # Evaluate the original callable.
        if scalar:
            return raw_gamma(float(raw_t[0]))
        return raw_gamma(raw_t)

    return _eval


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class Path3D:
    """A parameterized curve in 3D.

    Do not instantiate directly; use the class-method constructors:

    - :meth:`Path3D.from_callable`
    - :meth:`Path3D.from_polyline`
    - :meth:`Path3D.from_points`

    Parameters
    ----------
    _evaluator:
        Internal callable ``(t) → (3,) or (n, 3)``, ``t ∈ [0, 1]``.
    _arclength:
        Total arc length (float).
    _mode:
        Construction mode label (``'callable'``, ``'polyline'``, or
        ``'point-sample'``).
    _domain:
        Optional :class:`~pygeoinf3d.core.domain_base.Domain3D` reference.
    """

    def __init__(
        self,
        _evaluator: Callable,
        _arclength: float,
        _mode: str,
        _domain=None,
    ) -> None:
        self._evaluator = _evaluator
        self._arclength = float(_arclength)
        self._mode = _mode
        self._domain = _domain

    # ------------------------------------------------------------------
    # Factory constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_callable(
        cls,
        gamma: Callable,
        t_start: float,
        t_end: float,
        *,
        domain=None,
        n_approx_pts: int = 200,
    ) -> "Path3D":
        """Construct a path from a smooth parameterization.

        Parameters
        ----------
        gamma:
            Callable ``gamma(t)`` where scalar *t* → ``(3,)`` and
            array ``t`` of shape ``(n,)`` → ``(n, 3)``.
        t_start, t_end:
            Parameter domain of *gamma*.
        domain:
            Optional :class:`~pygeoinf3d.core.domain_base.Domain3D`.
        n_approx_pts:
            Number of uniform samples used to approximate the arc length
            and to build the internal arc-length re-parameterization.

        Returns
        -------
        Path3D
        """
        arclength = _approx_arclength_callable(
            gamma, t_start, t_end, n_approx_pts
        )
        evaluator = _build_callable_evaluator(
            gamma, t_start, t_end, arclength, n_approx_pts
        )
        return cls(evaluator, arclength, "callable", domain)

    @classmethod
    def from_polyline(
        cls,
        waypoints,
        *,
        domain=None,
    ) -> "Path3D":
        """Construct a path from an ordered array of waypoints.

        The path is piecewise-linear, parameterized by normalized
        chord-length arc length.

        Parameters
        ----------
        waypoints:
            Array of shape ``(n, 3)``, ``n >= 2``.
        domain:
            Optional :class:`~pygeoinf3d.core.domain_base.Domain3D`.

        Returns
        -------
        Path3D
        """
        waypoints = np.asarray(waypoints, dtype=float)
        if waypoints.ndim != 2 or waypoints.shape[1] != 3:
            raise ValueError(
                f"waypoints must have shape (n, 3); got {waypoints.shape}."
            )
        if len(waypoints) < 2:
            raise ValueError("At least 2 waypoints are required.")

        evaluator, arclength = _build_polyline_evaluator(waypoints)
        return cls(evaluator, arclength, "polyline", domain)

    @classmethod
    def from_points(
        cls,
        points,
        *,
        domain=None,
    ) -> "Path3D":
        """Construct a path from an ordered array of sample points.

        In Phase 2 this is identical to :meth:`from_polyline`
        (piecewise-linear interpolation).  Spline interpolation is deferred.

        Parameters
        ----------
        points:
            Array of shape ``(n, 3)``, ``n >= 2``.
        domain:
            Optional :class:`~pygeoinf3d.core.domain_base.Domain3D`.

        Returns
        -------
        Path3D
        """
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(
                f"points must have shape (n, 3); got {points.shape}."
            )
        if len(points) < 2:
            raise ValueError("At least 2 points are required.")

        evaluator, arclength = _build_polyline_evaluator(points)
        return cls(evaluator, arclength, "point-sample", domain)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __call__(self, t):
        """Evaluate the path at normalized arc-length parameter *t*.

        Parameters
        ----------
        t:
            Scalar in ``[0, 1]`` → returns ``(3,)`` array.
            Array of shape ``(n,)`` → returns ``(n, 3)`` array.

        Returns
        -------
        np.ndarray
            Shape ``(3,)`` or ``(n, 3)``.
        """
        return self._evaluator(t)

    @property
    def arclength(self) -> float:
        """Total arc length of the path (float)."""
        return self._arclength

    @property
    def domain(self):
        """The :class:`~pygeoinf3d.core.domain_base.Domain3D` this path lives
        in, or ``None`` if not specified."""
        return self._domain

    def arc_length_nodes(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return *n* equally-spaced (in arc length) nodes.

        Parameters
        ----------
        n:
            Number of nodes.

        Returns
        -------
        t_arr : np.ndarray
            Shape ``(n,)`` of normalized parameter values in ``[0, 1]``.
        pts_arr : np.ndarray
            Shape ``(n, 3)`` of 3D coordinates.
        """
        t_arr = np.linspace(0.0, 1.0, n)
        pts_arr = self._evaluator(t_arr)
        return t_arr, pts_arr

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Path3D(mode='{self._mode}', arclength={self._arclength:.6g})"
        )
