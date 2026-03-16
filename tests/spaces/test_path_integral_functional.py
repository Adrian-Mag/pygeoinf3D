"""Tests for PathIntegralFunctional.

Covers:
- Construction and basic properties
- Evaluation with midpoint-rule quadrature
- Linear-form algebra (scalar mul, neg, add, sub)
- Domain validation
- Different-path independence
"""

from __future__ import annotations

import pytest
import numpy as np
from typing import Any

from pygeoinf import EuclideanSpace, Ball

from pygeoinf3d.core.config import AxisAlignedBoundingBox
from pygeoinf3d.core.region import Region3D
from pygeoinf3d.core.domain_base import Domain3D
from pygeoinf3d.core.boundary import Boundary3D
from pygeoinf3d.core.functions import Function3D
from pygeoinf3d.core.paths import Path3D
from pygeoinf3d.spaces.lebesgue import Lebesgue3D
from pygeoinf3d.spaces.forms import VolumeKernel3D, PathIntegralFunctional


# ---------------------------------------------------------------------------
# Minimal mock domain (no integration needed for path tests)
# ---------------------------------------------------------------------------

class _MockBoundary(Boundary3D):
    def contains(self, point: np.ndarray) -> bool:
        return False


class _MockDomain(Domain3D):
    """Minimal Domain3D for path-integral tests; integration not exercised."""

    def sample_interior(self, n: int, *, rng: Any = None) -> np.ndarray:
        rng = np.random.default_rng(rng)
        return rng.uniform(-1.0, 1.0, size=(n, 3))

    def integrate_volume(self, f, *, method=None, **kwargs) -> float:
        n = kwargs.get("n", 10_000)
        seed = kwargs.get("seed", 42)
        pts = self.sample_interior(n, rng=seed)
        vals = f(pts)
        vol = 8.0  # [−1,1]^3
        return float(np.mean(vals)) * vol

    @property
    def boundary(self) -> Boundary3D:
        return _MockBoundary(self)


def _make_region() -> Region3D:
    space = EuclideanSpace(3)
    ball = Ball(space, np.zeros(3), 2.0)
    bbox = AxisAlignedBoundingBox(low=-np.ones(3), high=np.ones(3))
    return Region3D(ball, bounding_box=bbox)


def _make_domain() -> _MockDomain:
    return _MockDomain(_make_region())


def _make_space() -> Lebesgue3D:
    return Lebesgue3D(_make_domain())


def _make_domain_and_space():
    dom = _make_domain()
    return dom, Lebesgue3D(dom)


# ---------------------------------------------------------------------------
# Path fixtures
# ---------------------------------------------------------------------------

def _straight_path(x0=0.0, x1=1.0) -> Path3D:
    """Straight line from (x0,0,0) to (x1,0,0)."""
    return Path3D.from_polyline(
        np.array([[x0, 0.0, 0.0], [x1, 0.0, 0.0]])
    )


def _circular_arc(R: float, theta: float) -> Path3D:
    """Circular arc of radius R sweeping angle theta in the x-y plane."""
    angles = np.linspace(0.0, theta, 200)
    pts = np.column_stack([
        R * np.cos(angles),
        R * np.sin(angles),
        np.zeros(200),
    ])
    return Path3D.from_polyline(pts)


# ---------------------------------------------------------------------------
# Function fixtures
# ---------------------------------------------------------------------------

def _const_fn(domain, c: float) -> Function3D:
    return Function3D(domain, evaluate_callable=lambda pts: np.full(len(pts), c))


def _linear_x_fn(domain) -> Function3D:
    """f(x,y,z) = x."""
    return Function3D(domain, evaluate_callable=lambda pts: pts[:, 0])


def _sin_fn(domain) -> Function3D:
    """f(x,y,z) = sin(pi*x) — nontrivial for accuracy tests."""
    return Function3D(
        domain,
        evaluate_callable=lambda pts: np.sin(np.pi * pts[:, 0]),
    )


# ===========================================================================
# Constructor tests
# ===========================================================================

class TestPathIntegralFunctionalConstructor:

    def test_construction_straight_line(self):
        from pygeoinf import LinearForm
        dom, space = _make_domain_and_space()
        path = _straight_path()
        ell = PathIntegralFunctional(space, path)
        assert isinstance(ell, LinearForm)

    def test_domain_is_lebesgue_space(self):
        dom, space = _make_domain_and_space()
        path = _straight_path()
        ell = PathIntegralFunctional(space, path)
        assert ell.domain is space

    def test_result_is_scalar(self):
        """Evaluating the functional returns a Python float (scalar)."""
        dom, space = _make_domain_and_space()
        path = _straight_path()
        f = _const_fn(dom, 1.0)
        ell = PathIntegralFunctional(space, path)
        result = ell(f)
        assert isinstance(result, float), f"Expected float, got {type(result)}"

    def test_path_stored(self):
        dom, space = _make_domain_and_space()
        path = _straight_path()
        ell = PathIntegralFunctional(space, path)
        assert ell.path is path

    def test_n_quad_default(self):
        dom, space = _make_domain_and_space()
        path = _straight_path()
        ell = PathIntegralFunctional(space, path)
        assert ell.n_quad == 200

    def test_n_quad_custom(self):
        dom, space = _make_domain_and_space()
        path = _straight_path()
        ell = PathIntegralFunctional(space, path, n_quad=50)
        assert ell.n_quad == 50

    def test_domain_mismatch_raises(self):
        """Path with a *different* Domain3D than the space should raise."""
        dom1, space = _make_domain_and_space()
        dom2 = _make_domain()  # different object
        path = Path3D.from_polyline(
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            domain=dom2,
        )
        with pytest.raises(ValueError):
            PathIntegralFunctional(space, path)

    def test_bad_space_raises_typeerror(self):
        """Passing a non-Lebesgue3D as space raises TypeError."""
        path = _straight_path()
        with pytest.raises(TypeError, match="Lebesgue3D"):
            PathIntegralFunctional(object(), path)

    def test_bad_path_raises_typeerror(self):
        """Passing a non-Path3D as path raises TypeError."""
        dom, space = _make_domain_and_space()
        with pytest.raises(TypeError, match="Path3D"):
            PathIntegralFunctional(space, object())

    def test_components_raises_typeerror(self):
        dom, space = _make_domain_and_space()
        path = _straight_path()
        ell = PathIntegralFunctional(space, path)
        with pytest.raises(TypeError):
            _ = ell.components


# ===========================================================================
# Evaluation tests
# ===========================================================================

class TestPathIntegralFunctionalEvaluation:

    def test_constant_function_straight_line(self):
        """ℓ_γ(c) ≈ c × arclength = c × 1 for unit straight line."""
        c = 3.0
        dom, space = _make_domain_and_space()
        path = _straight_path(0.0, 1.0)  # length = 1
        f = _const_fn(dom, c)
        ell = PathIntegralFunctional(space, path, n_quad=200)
        result = ell(f)
        np.testing.assert_allclose(result, c * 1.0, rtol=1e-2)

    def test_constant_function_arclength_scaling(self):
        """ℓ_γ(c) ≈ c × L for path of length L."""
        c = 2.5
        L = 5.0
        dom, space = _make_domain_and_space()
        path = _straight_path(0.0, L)
        f = _const_fn(dom, c)
        ell = PathIntegralFunctional(space, path, n_quad=200)
        result = ell(f)
        np.testing.assert_allclose(result, c * L, rtol=1e-2)

    def test_linear_function_circle(self):
        """For f=1 and a circular arc of radius R and angle θ, ℓ_γ = R*θ."""
        R = 2.0
        theta = np.pi / 2
        dom, space = _make_domain_and_space()
        path = _circular_arc(R, theta)
        f = _const_fn(dom, 1.0)
        ell = PathIntegralFunctional(space, path, n_quad=500)
        expected = R * theta
        result = ell(f)
        np.testing.assert_allclose(result, expected, rtol=1e-2)

    def test_pointwise_scalar_correct(self):
        """For f(x,y,z)=x and path (0,0,0)→(2,0,0): ℓ = ∫₀² x dx = 2."""
        dom, space = _make_domain_and_space()
        path = _straight_path(0.0, 2.0)  # length = 2
        f = _linear_x_fn(dom)
        ell = PathIntegralFunctional(space, path, n_quad=500)
        # ∫₀² x dx = [x²/2]₀² = 2
        np.testing.assert_allclose(ell(f), 2.0, rtol=1e-2)

    def test_higher_n_quad_more_accurate(self):
        """Higher n_quad gives a smaller absolute error for sin(πx) on [0,1]."""
        dom, space = _make_domain_and_space()
        path = _straight_path(0.0, 1.0)
        f = _sin_fn(dom)
        # Analytic: ∫₀¹ sin(πx) dx = 2/π
        exact = 2.0 / np.pi
        ell_lo = PathIntegralFunctional(space, path, n_quad=5)
        ell_hi = PathIntegralFunctional(space, path, n_quad=500)
        err_lo = abs(ell_lo(f) - exact)
        err_hi = abs(ell_hi(f) - exact)
        assert err_hi < err_lo, (
            f"Higher n_quad should be more accurate: "
            f"err_lo={err_lo:.4f}, err_hi={err_hi:.4f}"
        )

    def test_n_quad_1_raises(self):
        """n_quad=1 is invalid for composite trapezoidal rule."""
        dom, space = _make_domain_and_space()
        path = _straight_path(0.0, 1.0)
        with pytest.raises(ValueError, match="n_quad"):
            PathIntegralFunctional(space, path, n_quad=1)

    def test_n_quad_0_raises(self):
        """n_quad=0 is invalid."""
        dom, space = _make_domain_and_space()
        path = _straight_path(0.0, 1.0)
        with pytest.raises(ValueError, match="n_quad"):
            PathIntegralFunctional(space, path, n_quad=0)


# ===========================================================================
# Algebra tests
# ===========================================================================

class TestPathIntegralFunctionalAlgebra:

    def _setup(self):
        dom, space = _make_domain_and_space()
        path = _straight_path(0.0, 1.0)
        f = _const_fn(dom, 3.0)
        ell = PathIntegralFunctional(space, path, n_quad=100)
        return dom, space, path, f, ell

    def test_scale_mul(self):
        dom, space, path, f, ell = self._setup()
        base = ell(f)
        scaled = (2.0 * ell)(f)
        np.testing.assert_allclose(scaled, 2.0 * base, rtol=1e-10)

    def test_scale_rmul(self):
        dom, space, path, f, ell = self._setup()
        base = ell(f)
        scaled = (ell * 3.0)(f)
        np.testing.assert_allclose(scaled, 3.0 * base, rtol=1e-10)

    def test_neg(self):
        dom, space, path, f, ell = self._setup()
        base = ell(f)
        neg = (-ell)(f)
        np.testing.assert_allclose(neg, -base, rtol=1e-10)

    def test_truediv(self):
        dom, space, path, f, ell = self._setup()
        base = ell(f)
        divided = (ell / 4.0)(f)
        np.testing.assert_allclose(divided, base / 4.0, rtol=1e-10)

    def test_add_same_path(self):
        dom, space, path, f, ell = self._setup()
        base = ell(f)
        added = (ell + ell)(f)
        np.testing.assert_allclose(added, 2.0 * base, rtol=1e-10)

    def test_sub_same_path(self):
        dom, space, path, f, ell = self._setup()
        diff = (ell - ell)(f)
        np.testing.assert_allclose(diff, 0.0, atol=1e-10)

    def test_add_typeerror_on_volume_kernel(self):
        dom, space, path, f, ell = self._setup()
        kernel = VolumeKernel3D(space, f)
        with pytest.raises(TypeError):
            _ = ell + kernel

    def test_linearity_on_functions(self):
        """\u2113(α*f + β*g) == α*ℓ(f) + β*ℓ(g) to floating-point tolerance."""
        dom, space = _make_domain_and_space()
        path = _straight_path(0.0, 1.0)
        ell = PathIntegralFunctional(space, path, n_quad=200)
        alpha, beta = 2.0, -1.5
        f = _const_fn(dom, 3.0)
        g = _linear_x_fn(dom)
        # Build α*f + β*g directly as a new callable
        fg = Function3D(
            dom,
            evaluate_callable=lambda pts: (
                alpha * np.full(len(pts), 3.0) + beta * pts[:, 0]
            ),
        )
        lhs = ell(fg)
        rhs = alpha * ell(f) + beta * ell(g)
        np.testing.assert_allclose(lhs, rhs, rtol=1e-10)


# ===========================================================================
# Different-path tests
# ===========================================================================

class TestPathIntegralFunctionalDifferentPaths:

    def test_different_paths_same_space(self):
        """Two functionals on different paths of the same space both work."""
        dom, space = _make_domain_and_space()
        path_a = _straight_path(0.0, 1.0)
        path_b = _straight_path(0.0, 2.0)
        f = _const_fn(dom, 5.0)
        ell_a = PathIntegralFunctional(space, path_a, n_quad=200)
        ell_b = PathIntegralFunctional(space, path_b, n_quad=200)
        np.testing.assert_allclose(ell_a(f), 5.0 * 1.0, rtol=1e-2)
        np.testing.assert_allclose(ell_b(f), 5.0 * 2.0, rtol=1e-2)

    def test_polyline_path_consistent_arclength(self):
        """Polyline path from (0,0,0)→(1,0,0)→(1,1,0) has arclength ≈ 2."""
        dom, space = _make_domain_and_space()
        waypoints = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
        path = Path3D.from_polyline(waypoints)
        f = _const_fn(dom, 1.0)
        ell = PathIntegralFunctional(space, path, n_quad=500)
        # arclength = 1 + 1 = 2; ℓ(1) ≈ 2
        np.testing.assert_allclose(ell(f), 2.0, rtol=1e-2)


# ===========================================================================
# Domain-validation tests
# ===========================================================================

class TestPathIntegralFunctionalDomainValidation:

    def test_none_domain_path_always_accepted(self):
        """Path3D with domain=None is accepted by any Lebesgue3D space."""
        dom, space = _make_domain_and_space()
        path = Path3D.from_polyline(
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            domain=None,
        )
        # Should not raise
        ell = PathIntegralFunctional(space, path)
        assert ell.path is path

    def test_matching_domain_path_accepted(self):
        """Path with domain matching space.function_domain is accepted."""
        dom, space = _make_domain_and_space()
        path = Path3D.from_polyline(
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            domain=dom,
        )
        # Should not raise
        ell = PathIntegralFunctional(space, path)
        assert ell.path is path
