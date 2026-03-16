"""Tests for pygeoinf3d.core.paths.Path3D.

Covers:
- from_callable: shape, arclength approximation, arc_length_nodes
- from_polyline: endpoint evaluation, midpoint, arclength, node count
- from_points: same-as-polyline and endpoint checks
- General: domain storage, domain=None default, repr
"""

from __future__ import annotations

import numpy as np
import pytest

from pygeoinf3d.core.paths import Path3D


# ---------------------------------------------------------------------------
# Tiny mock domain for .domain tests
# ---------------------------------------------------------------------------

class _MockDomain3D:
    dim = 3


# ---------------------------------------------------------------------------
# Callable path
# ---------------------------------------------------------------------------

class TestFromCallable:
    """Path3D.from_callable tests."""

    def _make_circle_path(self, radius: float = 1.0) -> Path3D:
        """Parameterize a full circle in the xy-plane."""
        def gamma(t):
            t = np.asarray(t, dtype=float)
            if t.ndim == 0:
                return np.array([radius * np.cos(t), radius * np.sin(t), 0.0])
            return np.stack(
                [radius * np.cos(t), radius * np.sin(t), np.zeros_like(t)],
                axis=1,
            )
        return Path3D.from_callable(gamma, 0.0, 2 * np.pi, n_approx_pts=1000)

    def test_from_callable_evaluates_correct_shape_scalar(self):
        path = self._make_circle_path()
        pt = path(0.0)
        assert isinstance(pt, np.ndarray)
        assert pt.shape == (3,)

    def test_from_callable_evaluates_correct_shape_array(self):
        path = self._make_circle_path()
        t_arr = np.linspace(0.0, 1.0, 7)
        pts = path(t_arr)
        assert isinstance(pts, np.ndarray)
        assert pts.shape == (7, 3)

    def test_from_callable_arclength_approximate(self):
        """Arclength of a full unit circle should be close to 2π."""
        path = self._make_circle_path(radius=1.0)
        np.testing.assert_allclose(path.arclength, 2 * np.pi, rtol=1e-3)

    def test_from_callable_arclength_larger_radius(self):
        """Arclength scales linearly with radius."""
        r = 3.0
        path = self._make_circle_path(radius=r)
        np.testing.assert_allclose(path.arclength, 2 * np.pi * r, rtol=1e-3)

    def test_from_callable_arc_length_nodes_count(self):
        path = self._make_circle_path()
        n = 11
        t_arr, pts_arr = path.arc_length_nodes(n)
        assert t_arr.shape == (n,)
        assert pts_arr.shape == (n, 3)

    def test_from_callable_arc_length_nodes_t_in_unit(self):
        """t values returned by arc_length_nodes must lie in [0, 1]."""
        path = self._make_circle_path()
        t_arr, _pts = path.arc_length_nodes(20)
        assert float(t_arr[0]) == pytest.approx(0.0, abs=1e-10)
        assert float(t_arr[-1]) == pytest.approx(1.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Polyline path
# ---------------------------------------------------------------------------

class TestFromPolyline:
    """Path3D.from_polyline tests."""

    def _straight_line(self) -> Path3D:
        """Straight line from origin to (3, 0, 0)."""
        pts = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        return Path3D.from_polyline(pts)

    def _l_polyline(self) -> Path3D:
        """Two equal-length segments: (0,0,0)→(1,0,0)→(1,1,0)."""
        pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
        return Path3D.from_polyline(pts)

    def test_from_polyline_evaluates_at_first_waypoint(self):
        path = self._straight_line()
        np.testing.assert_allclose(path(0.0), [0.0, 0.0, 0.0], atol=1e-10)

    def test_from_polyline_evaluates_at_last_waypoint(self):
        path = self._straight_line()
        np.testing.assert_allclose(path(1.0), [3.0, 0.0, 0.0], atol=1e-10)

    def test_from_polyline_midpoint_straight(self):
        """Midpoint of a straight line is the midpoint coordinate."""
        path = self._straight_line()
        np.testing.assert_allclose(path(0.5), [1.5, 0.0, 0.0], atol=1e-10)

    def test_from_polyline_midpoint_l_shape(self):
        """For equal-length L-polyline, t=0.5 is exactly at the corner."""
        path = self._l_polyline()
        np.testing.assert_allclose(path(0.5), [1.0, 0.0, 0.0], atol=1e-10)

    def test_from_polyline_arclength_exact(self):
        """Arclength of straight line (0,0,0)->(3,0,0) is exactly 3.0."""
        path = self._straight_line()
        np.testing.assert_allclose(path.arclength, 3.0, atol=1e-14)

    def test_from_polyline_arclength_l_shape(self):
        """Arclength of L-polyline with two unit segments is 2.0."""
        path = self._l_polyline()
        np.testing.assert_allclose(path.arclength, 2.0, atol=1e-14)

    def test_from_polyline_arc_length_nodes_count(self):
        path = self._straight_line()
        n = 5
        t_arr, pts_arr = path.arc_length_nodes(n)
        assert t_arr.shape == (n,)
        assert pts_arr.shape == (n, 3)

    def test_from_polyline_arc_length_nodes_endpoints(self):
        path = self._straight_line()
        t_arr, pts_arr = path.arc_length_nodes(5)
        np.testing.assert_allclose(pts_arr[0], [0.0, 0.0, 0.0], atol=1e-10)
        np.testing.assert_allclose(pts_arr[-1], [3.0, 0.0, 0.0], atol=1e-10)

    def test_from_polyline_batch_evaluation(self):
        path = self._straight_line()
        t_arr = np.array([0.0, 0.5, 1.0])
        pts = path(t_arr)
        assert pts.shape == (3, 3)
        np.testing.assert_allclose(pts[0], [0.0, 0.0, 0.0], atol=1e-10)
        np.testing.assert_allclose(pts[-1], [3.0, 0.0, 0.0], atol=1e-10)


# ---------------------------------------------------------------------------
# Point-sample path
# ---------------------------------------------------------------------------

class TestFromPoints:
    """Path3D.from_points tests (same-as-polyline for Phase 2)."""

    def _make_pts(self) -> np.ndarray:
        return np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
        ])

    def test_from_points_evaluates_at_first_endpoint(self):
        pts = self._make_pts()
        path = Path3D.from_points(pts)
        np.testing.assert_allclose(path(0.0), pts[0], atol=1e-10)

    def test_from_points_evaluates_at_last_endpoint(self):
        pts = self._make_pts()
        path = Path3D.from_points(pts)
        np.testing.assert_allclose(path(1.0), pts[-1], atol=1e-10)

    def test_from_points_same_as_polyline_pointwise(self):
        """from_points and from_polyline must agree at a grid of t values."""
        pts = self._make_pts()
        path_poly = Path3D.from_polyline(pts)
        path_samp = Path3D.from_points(pts)
        t_grid = np.linspace(0.0, 1.0, 31)
        np.testing.assert_allclose(
            path_poly(t_grid), path_samp(t_grid), atol=1e-12
        )

    def test_from_points_arclength_matches_polyline(self):
        pts = self._make_pts()
        path_poly = Path3D.from_polyline(pts)
        path_samp = Path3D.from_points(pts)
        np.testing.assert_allclose(
            path_poly.arclength, path_samp.arclength, atol=1e-14
        )


# ---------------------------------------------------------------------------
# General / metadata tests
# ---------------------------------------------------------------------------

class TestGeneral:
    """Cross-mode and metadata tests."""

    def test_domain_stored(self):
        dom = _MockDomain3D()
        pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        path = Path3D.from_polyline(pts, domain=dom)
        assert path.domain is dom

    def test_domain_none_by_default(self):
        pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        path = Path3D.from_polyline(pts)
        assert path.domain is None

    def test_repr_contains_mode_polyline(self):
        pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        path = Path3D.from_polyline(pts)
        r = repr(path)
        assert "polyline" in r.lower()
        # repr should show a float arclength
        assert "1.0" in r or "arc" in r.lower()

    def test_repr_contains_mode_callable(self):
        def gamma(t):
            t = np.asarray(t)
            if t.ndim == 0:
                return np.array([float(t), 0.0, 0.0])
            return np.stack([t, np.zeros_like(t), np.zeros_like(t)], axis=1)
        path = Path3D.from_callable(gamma, 0.0, 1.0)
        r = repr(path)
        assert "callable" in r.lower()

    def test_repr_contains_mode_points(self):
        pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        path = Path3D.from_points(pts)
        r = repr(path)
        assert "point" in r.lower()

    def test_callable_domain_stored(self):
        dom = _MockDomain3D()

        def gamma(t):
            t = np.asarray(t)
            if t.ndim == 0:
                return np.array([float(t), 0.0, 0.0])
            return np.stack([t, np.zeros_like(t), np.zeros_like(t)], axis=1)
        path = Path3D.from_callable(gamma, 0.0, 1.0, domain=dom)
        assert path.domain is dom
