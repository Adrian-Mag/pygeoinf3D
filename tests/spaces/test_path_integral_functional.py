"""Tests for PathIntegralFunctional.

Covers:
- Construction and basic properties
- Evaluation with midpoint-rule quadrature
- Linear-form algebra (scalar mul, neg, add, sub)
- Domain validation
- Different-path independence

Also covers SumLinearForm (Phase 4 Revision):
- Construction and evaluation
- Weighted-sum semantics
- Components access raises TypeError
- Domain validation
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
from pygeoinf3d.spaces.forms import (
    VolumeKernel3D,
    PathIntegralFunctional,
    SumLinearForm,
)
from pygeoinf3d.spaces.sobolev import Sobolev3D
from pygeoinf import LinearForm


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


def _make_sobolev_space() -> Sobolev3D:
    """Return a Sobolev3D with identity weight (W = I) for testing."""
    from pygeoinf.linear_operators import LinearOperator
    dom = _make_domain()
    lebesgue = Lebesgue3D(dom)

    def _identity(f):
        return f

    W = LinearOperator(lebesgue, lebesgue, _identity, adjoint_mapping=_identity)
    return Sobolev3D(lebesgue, W, W)


def _make_domain_and_sobolev():
    """Return (domain, Sobolev3D) with identity weight for testing."""
    from pygeoinf.linear_operators import LinearOperator
    dom = _make_domain()
    lebesgue = Lebesgue3D(dom)

    def _identity(f):
        return f

    W = LinearOperator(lebesgue, lebesgue, _identity, adjoint_mapping=_identity)
    sobolev = Sobolev3D(lebesgue, W, W)
    return dom, sobolev


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
        dom, sobolev = _make_domain_and_sobolev()
        path = _straight_path()
        ell = PathIntegralFunctional(sobolev, path)
        assert isinstance(ell, LinearForm)

    def test_domain_is_sobolev_space(self):
        dom, sobolev = _make_domain_and_sobolev()
        path = _straight_path()
        ell = PathIntegralFunctional(sobolev, path)
        assert ell.domain is sobolev

    def test_result_is_scalar(self):
        """Evaluating the functional returns a Python float (scalar)."""
        dom, sobolev = _make_domain_and_sobolev()
        path = _straight_path()
        f = _const_fn(dom, 1.0)
        ell = PathIntegralFunctional(sobolev, path)
        result = ell(f)
        assert isinstance(result, float), f"Expected float, got {type(result)}"

    def test_path_stored(self):
        dom, sobolev = _make_domain_and_sobolev()
        path = _straight_path()
        ell = PathIntegralFunctional(sobolev, path)
        assert ell.path is path

    def test_n_quad_default(self):
        dom, sobolev = _make_domain_and_sobolev()
        path = _straight_path()
        ell = PathIntegralFunctional(sobolev, path)
        assert ell.n_quad == 200

    def test_n_quad_custom(self):
        dom, sobolev = _make_domain_and_sobolev()
        path = _straight_path()
        ell = PathIntegralFunctional(sobolev, path, n_quad=50)
        assert ell.n_quad == 50

    def test_domain_mismatch_raises(self):
        """Path with a *different* Domain3D than the space should raise."""
        dom1, sobolev = _make_domain_and_sobolev()
        dom2 = _make_domain()  # different object
        path = Path3D.from_polyline(
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            domain=dom2,
        )
        with pytest.raises(ValueError):
            PathIntegralFunctional(sobolev, path)

    def test_bad_space_raises_typeerror(self):
        """Passing a non-Sobolev3D as space raises TypeError."""
        path = _straight_path()
        with pytest.raises(TypeError, match="Sobolev3D"):
            PathIntegralFunctional(object(), path)

    def test_lebesgue3d_space_raises_typeerror(self):
        """Passing a Lebesgue3D (not Sobolev3D) as space raises TypeError."""
        dom, lebesgue = _make_domain_and_space()
        path = _straight_path()
        with pytest.raises(TypeError, match="Sobolev3D"):
            PathIntegralFunctional(lebesgue, path)

    def test_bad_path_raises_typeerror(self):
        """Passing a non-Path3D as path raises TypeError."""
        dom, sobolev = _make_domain_and_sobolev()
        with pytest.raises(TypeError, match="Path3D"):
            PathIntegralFunctional(sobolev, object())

    def test_components_raises_typeerror(self):
        dom, sobolev = _make_domain_and_sobolev()
        path = _straight_path()
        ell = PathIntegralFunctional(sobolev, path)
        with pytest.raises(TypeError):
            _ = ell.components


# ===========================================================================
# Evaluation tests
# ===========================================================================

class TestPathIntegralFunctionalEvaluation:

    def test_constant_function_straight_line(self):
        """ℓ_γ(c) ≈ c × arclength = c × 1 for unit straight line."""
        c = 3.0
        dom, sobolev = _make_domain_and_sobolev()
        path = _straight_path(0.0, 1.0)  # length = 1
        f = _const_fn(dom, c)
        ell = PathIntegralFunctional(sobolev, path, n_quad=200)
        result = ell(f)
        np.testing.assert_allclose(result, c * 1.0, rtol=1e-2)

    def test_constant_function_arclength_scaling(self):
        """ℓ_γ(c) ≈ c × L for path of length L."""
        c = 2.5
        L = 5.0
        dom, sobolev = _make_domain_and_sobolev()
        path = _straight_path(0.0, L)
        f = _const_fn(dom, c)
        ell = PathIntegralFunctional(sobolev, path, n_quad=200)
        result = ell(f)
        np.testing.assert_allclose(result, c * L, rtol=1e-2)

    def test_linear_function_circle(self):
        """For f=1 and a circular arc of radius R and angle θ, ℓ_γ = R*θ."""
        R = 2.0
        theta = np.pi / 2
        dom, sobolev = _make_domain_and_sobolev()
        path = _circular_arc(R, theta)
        f = _const_fn(dom, 1.0)
        ell = PathIntegralFunctional(sobolev, path, n_quad=500)
        expected = R * theta
        result = ell(f)
        np.testing.assert_allclose(result, expected, rtol=1e-2)

    def test_pointwise_scalar_correct(self):
        """For f(x,y,z)=x and path (0,0,0)→(2,0,0): ℓ = ∫₀² x dx = 2."""
        dom, sobolev = _make_domain_and_sobolev()
        path = _straight_path(0.0, 2.0)  # length = 2
        f = _linear_x_fn(dom)
        ell = PathIntegralFunctional(sobolev, path, n_quad=500)
        # ∫₀² x dx = [x²/2]₀² = 2
        np.testing.assert_allclose(ell(f), 2.0, rtol=1e-2)

    def test_higher_n_quad_more_accurate(self):
        """Higher n_quad gives a smaller absolute error for sin(πx) on [0,1]."""
        dom, sobolev = _make_domain_and_sobolev()
        path = _straight_path(0.0, 1.0)
        f = _sin_fn(dom)
        # Analytic: ∫₀¹ sin(πx) dx = 2/π
        exact = 2.0 / np.pi
        ell_lo = PathIntegralFunctional(sobolev, path, n_quad=5)
        ell_hi = PathIntegralFunctional(sobolev, path, n_quad=500)
        err_lo = abs(ell_lo(f) - exact)
        err_hi = abs(ell_hi(f) - exact)
        assert err_hi < err_lo, (
            f"Higher n_quad should be more accurate: "
            f"err_lo={err_lo:.4f}, err_hi={err_hi:.4f}"
        )

    def test_n_quad_1_raises(self):
        """n_quad=1 is invalid for composite trapezoidal rule."""
        dom, sobolev = _make_domain_and_sobolev()
        path = _straight_path(0.0, 1.0)
        with pytest.raises(ValueError, match="n_quad"):
            PathIntegralFunctional(sobolev, path, n_quad=1)

    def test_n_quad_0_raises(self):
        """n_quad=0 is invalid."""
        dom, sobolev = _make_domain_and_sobolev()
        path = _straight_path(0.0, 1.0)
        with pytest.raises(ValueError, match="n_quad"):
            PathIntegralFunctional(sobolev, path, n_quad=0)


# ===========================================================================
# Algebra tests
# ===========================================================================

class TestPathIntegralFunctionalAlgebra:

    def _setup(self):
        dom, sobolev = _make_domain_and_sobolev()
        path = _straight_path(0.0, 1.0)
        f = _const_fn(dom, 3.0)
        ell = PathIntegralFunctional(sobolev, path, n_quad=100)
        return dom, sobolev, path, f, ell

    def test_scale_mul(self):
        dom, sobolev, path, f, ell = self._setup()
        base = ell(f)
        scaled = (2.0 * ell)(f)
        np.testing.assert_allclose(scaled, 2.0 * base, rtol=1e-10)

    def test_scale_rmul(self):
        dom, sobolev, path, f, ell = self._setup()
        base = ell(f)
        scaled = (ell * 3.0)(f)
        np.testing.assert_allclose(scaled, 3.0 * base, rtol=1e-10)

    def test_neg(self):
        dom, sobolev, path, f, ell = self._setup()
        base = ell(f)
        neg = (-ell)(f)
        np.testing.assert_allclose(neg, -base, rtol=1e-10)

    def test_truediv(self):
        dom, sobolev, path, f, ell = self._setup()
        base = ell(f)
        divided = (ell / 4.0)(f)
        np.testing.assert_allclose(divided, base / 4.0, rtol=1e-10)

    def test_add_same_path(self):
        dom, sobolev, path, f, ell = self._setup()
        base = ell(f)
        added = (ell + ell)(f)
        np.testing.assert_allclose(added, 2.0 * base, rtol=1e-10)

    def test_sub_same_path(self):
        dom, sobolev, path, f, ell = self._setup()
        diff = (ell - ell)(f)
        np.testing.assert_allclose(diff, 0.0, atol=1e-10)

    def test_add_typeerror_on_volume_kernel(self):
        dom, sobolev, path, f, ell = self._setup()
        # VolumeKernel3D still requires Lebesgue3D; build on the base_space
        kernel = VolumeKernel3D(sobolev.base_space, f)
        with pytest.raises(TypeError):
            _ = ell + kernel

    def test_linearity_on_functions(self):
        """ℓ(α*f + β*g) == α*ℓ(f) + β*ℓ(g) to floating-point tolerance."""
        dom, sobolev = _make_domain_and_sobolev()
        path = _straight_path(0.0, 1.0)
        ell = PathIntegralFunctional(sobolev, path, n_quad=200)
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
        dom, sobolev = _make_domain_and_sobolev()
        path_a = _straight_path(0.0, 1.0)
        path_b = _straight_path(0.0, 2.0)
        f = _const_fn(dom, 5.0)
        ell_a = PathIntegralFunctional(sobolev, path_a, n_quad=200)
        ell_b = PathIntegralFunctional(sobolev, path_b, n_quad=200)
        np.testing.assert_allclose(ell_a(f), 5.0 * 1.0, rtol=1e-2)
        np.testing.assert_allclose(ell_b(f), 5.0 * 2.0, rtol=1e-2)

    def test_polyline_path_consistent_arclength(self):
        """Polyline path from (0,0,0)→(1,0,0)→(1,1,0) has arclength ≈ 2."""
        dom, sobolev = _make_domain_and_sobolev()
        waypoints = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
        path = Path3D.from_polyline(waypoints)
        f = _const_fn(dom, 1.0)
        ell = PathIntegralFunctional(sobolev, path, n_quad=500)
        # arclength = 1 + 1 = 2; ℓ(1) ≈ 2
        np.testing.assert_allclose(ell(f), 2.0, rtol=1e-2)


# ===========================================================================
# Domain-validation tests
# ===========================================================================

class TestPathIntegralFunctionalDomainValidation:

    def test_none_domain_path_always_accepted(self):
        """Path3D with domain=None is accepted by any Sobolev3D space."""
        dom, sobolev = _make_domain_and_sobolev()
        path = Path3D.from_polyline(
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            domain=None,
        )
        # Should not raise
        ell = PathIntegralFunctional(sobolev, path)
        assert ell.path is path

    def test_matching_domain_path_accepted(self):
        """Path with domain matching space.function_domain is accepted."""
        dom, sobolev = _make_domain_and_sobolev()
        path = Path3D.from_polyline(
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            domain=dom,
        )
        # Should not raise
        ell = PathIntegralFunctional(sobolev, path)
        assert ell.path is path


# ---------------------------------------------------------------------------
# TestSumLinearForm  (Phase 4 Revision)
# ---------------------------------------------------------------------------

def _make_two_path_forms():
    """Return (sobolev, form_a, form_b) on a fresh Sobolev3D."""
    sobolev = _make_sobolev_space()
    path_a = Path3D.from_polyline(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    )
    path_b = Path3D.from_polyline(
        np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    )
    form_a = PathIntegralFunctional(sobolev, path_a)
    form_b = PathIntegralFunctional(sobolev, path_b)
    return sobolev, form_a, form_b


class TestSumLinearForm:
    """SumLinearForm: construction, evaluation, and validation."""

    def test_construction_valid(self):
        """Basic construction with two PathIntegralFunctional terms."""
        space, form_a, form_b = _make_two_path_forms()
        slm = SumLinearForm(space, [(1.0, form_a), (1.0, form_b)])
        assert isinstance(slm, LinearForm)
        assert slm.domain is space

    def test_evaluation_sum(self):
        """SumLinearForm([(1, ℓ1), (1, ℓ2)])(f) == ℓ1(f) + ℓ2(f)."""
        space, form_a, form_b = _make_two_path_forms()
        slm = SumLinearForm(space, [(1.0, form_a), (1.0, form_b)])

        f = _const_fn(space.function_domain, 1.0)
        expected = form_a(f) + form_b(f)
        result = slm(f)
        np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-10)

    def test_evaluation_weighted_sum(self):
        """SumLinearForm([(2, ℓ1), (-1, ℓ2)])(f) == 2*ℓ1(f) - ℓ2(f)."""
        space, form_a, form_b = _make_two_path_forms()
        slm = SumLinearForm(space, [(2.0, form_a), (-1.0, form_b)])

        f = _const_fn(space.function_domain, 3.0)
        expected = 2.0 * form_a(f) + (-1.0) * form_b(f)
        result = slm(f)
        np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-10)

    def test_components_raises(self):
        """Accessing .components raises TypeError."""
        space, form_a, form_b = _make_two_path_forms()
        slm = SumLinearForm(space, [(1.0, form_a), (1.0, form_b)])
        with pytest.raises(TypeError):
            _ = slm.components

    def test_terms_property(self):
        """.terms returns the list of (coeff, form) pairs."""
        space, form_a, form_b = _make_two_path_forms()
        terms_in = [(2.0, form_a), (-0.5, form_b)]
        slm = SumLinearForm(space, terms_in)
        terms_out = slm.terms
        assert len(terms_out) == 2
        assert terms_out[0][0] == pytest.approx(2.0)
        assert terms_out[0][1] is form_a
        assert terms_out[1][0] == pytest.approx(-0.5)
        assert terms_out[1][1] is form_b

    def test_domain_is_space(self):
        """.domain is the space passed at construction."""
        space, form_a, form_b = _make_two_path_forms()
        slm = SumLinearForm(space, [(1.0, form_a), (1.0, form_b)])
        assert slm.domain is space

    def test_domain_mismatch_raises(self):
        """Form defined on a different space raises ValueError."""
        space_a, form_a, _ = _make_two_path_forms()
        space_b = _make_sobolev_space()
        path_b = Path3D.from_polyline(
            np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        )
        form_b_wrong = PathIntegralFunctional(space_b, path_b)
        with pytest.raises(ValueError, match="same space"):
            SumLinearForm(space_a, [(1.0, form_a), (1.0, form_b_wrong)])

    def test_isinstance_linear_form(self):
        """SumLinearForm is a LinearForm instance."""
        space, form_a, form_b = _make_two_path_forms()
        slm = SumLinearForm(space, [(1.0, form_a), (1.0, form_b)])
        assert isinstance(slm, LinearForm)

    def test_terms_returns_copy(self):
        """.terms returns a new list so external mutation is safe."""
        space, form_a, form_b = _make_two_path_forms()
        slm = SumLinearForm(space, [(1.0, form_a), (1.0, form_b)])
        t1 = slm.terms
        t1.clear()
        assert len(slm.terms) == 2

    def test_evaluation_with_volume_kernel_terms(self):
        """SumLinearForm works with VolumeKernel3D terms."""
        from pygeoinf3d.core.functions import Function3D
        space = _make_space()
        dom = space.function_domain
        k1 = Function3D(dom, evaluate_callable=lambda pts: np.ones(len(pts)))
        k2 = Function3D(dom, evaluate_callable=lambda pts: 2.0 * np.ones(len(pts)))
        vk1 = VolumeKernel3D(space, k1)
        vk2 = VolumeKernel3D(space, k2)
        slm = SumLinearForm(space, [(1.0, vk1), (1.0, vk2)])
        f = _const_fn(dom, 1.0)
        # Each VolumeKernel3D evaluates f=1 against its kernel
        expected = vk1(f) + vk2(f)
        np.testing.assert_allclose(slm(f), expected, rtol=1e-10, atol=1e-10)


# ===========================================================================
# Caching tests (Phase 1 — vectorize-hot-paths)
# ===========================================================================

class TestPathIntegralFunctionalCaching:

    def test_path_integral_cached_nodes_same_as_direct(self):
        """Evaluation matches manually computed integral using arc_length_nodes."""
        dom, sobolev = _make_domain_and_sobolev()
        path = _straight_path(0.0, 1.0)
        ell = PathIntegralFunctional(sobolev, path, n_quad=50)
        f = _sin_fn(dom)

        # Manual computation using arc_length_nodes directly
        t_arr, pts = path.arc_length_nodes(50)
        vals = f(pts)
        expected = 1.0 * path.arclength * float(np.trapezoid(vals, t_arr))

        result = ell(f)
        np.testing.assert_allclose(result, expected, rtol=1e-14, atol=1e-14)

    def test_path_integral_caching_does_not_recompute(self):
        """After init, _pts and _t_arr exist; repeated calls give identical results."""
        dom, sobolev = _make_domain_and_sobolev()
        path = _straight_path()
        ell = PathIntegralFunctional(sobolev, path)

        # Cached arrays must exist after construction (Phase 1 requirement)
        assert hasattr(ell, "_pts"), "_pts must be cached in __init__"
        assert hasattr(ell, "_t_arr"), "_t_arr must be cached in __init__"

        # Repeated calls must return identical results
        f = _sin_fn(dom)
        r1 = ell(f)
        r2 = ell(f)
        assert r1 == r2, "Repeated evaluation must give identical results"

    def test_path_integral_copy_shares_path(self):
        """copy() returns a functional that gives the same result as the original."""
        dom, sobolev = _make_domain_and_sobolev()
        path = _straight_path()
        ell = PathIntegralFunctional(sobolev, path, n_quad=50)
        copied = ell.copy()
        f = _linear_x_fn(dom)
        np.testing.assert_allclose(copied(f), ell(f), rtol=1e-14, atol=1e-14)

    def test_path_integral_negation_works(self):
        """(-form)(f) == -form(f)."""
        dom, sobolev = _make_domain_and_sobolev()
        path = _straight_path()
        ell = PathIntegralFunctional(sobolev, path, n_quad=50)
        neg_ell = -ell
        f = _sin_fn(dom)
        np.testing.assert_allclose(neg_ell(f), -ell(f), rtol=1e-14, atol=1e-14)

    def test_path_integral_scalar_mul_works(self):
        """(2.0 * form)(f) == 2.0 * form(f)."""
        dom, sobolev = _make_domain_and_sobolev()
        path = _straight_path()
        ell = PathIntegralFunctional(sobolev, path, n_quad=50)
        scaled_ell = 2.0 * ell
        f = _sin_fn(dom)
        np.testing.assert_allclose(scaled_ell(f), 2.0 * ell(f), rtol=1e-14, atol=1e-14)

    def test_quad_weights_shape_and_dot_matches_eval(self):
        """quad_weights has shape (n_quad,) and dot(w, f(pts)) == ell(f)."""
        dom, sobolev = _make_domain_and_sobolev()
        path = _straight_path()
        n_quad = 80
        ell = PathIntegralFunctional(sobolev, path, n_quad=n_quad)

        assert ell.quad_weights.shape == (n_quad,), \
            "quad_weights must have shape (n_quad,)"

        f = _sin_fn(dom)
        via_dot = float(np.dot(ell.quad_weights, f(ell._pts)))
        via_call = float(ell(f))
        np.testing.assert_allclose(via_dot, via_call, rtol=1e-14, atol=1e-14)

    def test_quad_weights_consistent_under_scaling(self):
        """(a * form).quad_weights == a * form.quad_weights."""
        dom, sobolev = _make_domain_and_sobolev()
        path = _straight_path()
        ell = PathIntegralFunctional(sobolev, path, n_quad=60)
        a = 3.7
        scaled = a * ell
        np.testing.assert_allclose(
            scaled.quad_weights, a * ell.quad_weights, rtol=1e-14, atol=1e-14
        )
