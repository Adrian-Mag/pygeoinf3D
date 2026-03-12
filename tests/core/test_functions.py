"""Tests for pygeoinf3d.core.functions.Function3D.

Covers:
- Single-point and batched evaluation
- Domain-attachment mode detection
- Domain membership checks via domain.contains
- Arithmetic: +, -, scalar *, pointwise * (integrand formation)
- copy()
- attach_to_space() / detach() semantics
- Error paths (incompatible domains, missing callable, etc.)
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


# ---------------------------------------------------------------------------
# Evaluation helpers (module-level to keep lambdas under 79 chars)
# ---------------------------------------------------------------------------

def _x0(x):
    return x[:, 0] if x.ndim == 2 else float(x[0])


def _x1(x):
    return x[:, 1] if x.ndim == 2 else float(x[1])


def _const1(x):
    return np.ones(len(x)) if x.ndim == 2 else 1.0


# ---------------------------------------------------------------------------
# Minimal concrete domain for tests
# ---------------------------------------------------------------------------

class _MockBoundary(Boundary3D):
    def contains(self, point: np.ndarray) -> bool:
        return False


class _UnitBallDomain(Domain3D):
    """Concrete domain: unit ball centered at origin."""

    def sample_interior(self, n: int, *, rng: Any = None) -> np.ndarray:
        rng = np.random.default_rng(rng)
        pts = rng.standard_normal((n, 3))
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        r = rng.uniform(0, 1, size=(n, 1)) ** (1.0 / 3.0)
        return pts * r

    def integrate_volume(self, f, *, method=None, **kwargs) -> float:
        rng = np.random.default_rng(42)
        n = kwargs.get("n", 20_000)
        pts = self.sample_interior(n, rng=rng)
        vals = np.array([f(p) for p in pts])
        volume = (4.0 / 3.0) * np.pi
        return volume * float(np.mean(vals))

    @property
    def boundary(self) -> Boundary3D:
        return _MockBoundary(self)


def _make_ball_region() -> Region3D:
    space = EuclideanSpace(3)
    ball = Ball(space, np.zeros(3), 1.0)
    bbox = AxisAlignedBoundingBox(low=np.full(3, -1.0), high=np.full(3, 1.0))
    return Region3D(ball, bounding_box=bbox)


def _make_domain() -> _UnitBallDomain:
    return _UnitBallDomain(_make_ball_region())


# ---------------------------------------------------------------------------
# Helpers — lazy import to keep import failures close to test site
# ---------------------------------------------------------------------------

def _Function3D():
    from pygeoinf3d.core.functions import Function3D
    return Function3D


# ===========================================================================
# Construction
# ===========================================================================

class TestFunction3DConstruction:
    def test_standalone_construction_succeeds(self):
        F = _Function3D()
        dom = _make_domain()
        f = F(dom, evaluate_callable=lambda x: 1.0)
        assert f is not None

    def test_missing_callable_raises(self):
        F = _Function3D()
        dom = _make_domain()
        with pytest.raises((TypeError, ValueError)):
            F(dom)  # no callable

    def test_domain_attribute_standalone(self):
        F = _Function3D()
        dom = _make_domain()
        f = F(dom, evaluate_callable=lambda x: 0.0)
        assert f.domain is dom

    def test_space_is_none_standalone(self):
        F = _Function3D()
        dom = _make_domain()
        f = F(dom, evaluate_callable=lambda x: 0.0)
        assert f.space is None

    def test_is_attached_false(self):
        F = _Function3D()
        dom = _make_domain()
        f = F(dom, evaluate_callable=lambda x: 0.0)
        assert not f.is_attached


# ===========================================================================
# Evaluation
# ===========================================================================

class TestFunction3DEvaluation:
    def setup_method(self):
        F = _Function3D()
        dom = _make_domain()
        self.dom = dom
        # Callable handles both (3,) single-point and (n,3) batch input
        self.f = F(dom, evaluate_callable=_x0)

    def test_single_point_shape_3(self):
        pt = np.array([0.1, 0.2, 0.3])
        val = self.f(pt)
        assert np.isscalar(val) or np.shape(val) == ()
        np.testing.assert_allclose(np.asarray(val), 0.1, rtol=1e-12)

    def test_batched_eval_shape_n3(self):
        pts = np.array([[0.1, 0.0, 0.0], [0.5, 0.0, 0.0], [-0.3, 0.0, 0.0]])
        vals = np.asarray(self.f(pts))
        assert vals.shape == (3,), f"Expected (3,), got {vals.shape}"
        np.testing.assert_allclose(vals, [0.1, 0.5, -0.3], rtol=1e-12)

    def test_zero_callable(self):
        F = _Function3D()
        dom = _make_domain()
        g = F(dom, evaluate_callable=lambda x: 0.0)
        val = g(np.array([0.0, 0.0, 0.0]))
        np.testing.assert_allclose(float(val), 0.0, atol=1e-15)

    def test_constant_callable(self):
        F = _Function3D()
        dom = _make_domain()
        g = F(dom, evaluate_callable=_const1)
        pts = np.random.default_rng(7).uniform(-0.5, 0.5, (5, 3))
        vals = g(pts)
        np.testing.assert_allclose(vals, np.ones(5), rtol=1e-12)


# ===========================================================================
# Domain containment
# ===========================================================================

class TestFunction3DDomainContainment:
    def setup_method(self):
        F = _Function3D()
        dom = _make_domain()
        self.dom = dom
        self.f = F(dom, evaluate_callable=lambda x: 1.0)

    def test_domain_contains_interior(self):
        assert self.dom.contains(np.array([0.0, 0.0, 0.0]))

    def test_domain_excludes_exterior(self):
        assert not self.dom.contains(np.array([2.0, 0.0, 0.0]))

    def test_function_domain_is_domain(self):
        assert self.f.domain is self.dom

    def test_domain_check_for_boundary_point(self):
        # Points exactly on boundary — behavior must be consistent with domain
        pt = np.array([1.0, 0.0, 0.0])
        # Just verify it doesn't crash; membership depends on Ball's definition
        result = self.f.domain.contains(pt)
        assert isinstance(result, (bool, np.bool_))


# ===========================================================================
# Arithmetic
# ===========================================================================

class TestFunction3DArithmetic:
    def setup_method(self):
        F = _Function3D()
        dom = _make_domain()
        self.dom = dom
        # f(x) = x[0],  g(x) = x[1]
        self.f = F(dom, evaluate_callable=_x0)
        self.g = F(dom, evaluate_callable=_x1)

    def _eval_scalar(self, func, pt):
        return float(func(pt))

    def test_scalar_multiply(self):
        F = _Function3D()
        dom = _make_domain()
        f = F(dom, evaluate_callable=_x0)
        h = 3.0 * f
        pt = np.array([0.5, 0.0, 0.0])
        np.testing.assert_allclose(float(h(pt)), 1.5, rtol=1e-12)

    def test_scalar_multiply_right(self):
        F = _Function3D()
        dom = _make_domain()
        f = F(dom, evaluate_callable=_x0)
        h = f * 2.0
        pt = np.array([0.4, 0.0, 0.0])
        np.testing.assert_allclose(float(h(pt)), 0.8, rtol=1e-12)

    def test_addition(self):
        F = _Function3D()
        dom = _make_domain()
        f = F(dom, evaluate_callable=_x0)
        g = F(dom, evaluate_callable=_x1)
        h = f + g
        pt = np.array([0.3, 0.4, 0.0])
        np.testing.assert_allclose(float(h(pt)), 0.7, rtol=1e-12)

    def test_subtraction(self):
        F = _Function3D()
        dom = _make_domain()
        f = F(dom, evaluate_callable=_x0)
        g = F(dom, evaluate_callable=_x1)
        h = f - g
        pt = np.array([0.6, 0.2, 0.0])
        np.testing.assert_allclose(float(h(pt)), 0.4, rtol=1e-12)

    def test_negation(self):
        F = _Function3D()
        dom = _make_domain()
        f = F(dom, evaluate_callable=_x0)
        h = -f
        pt = np.array([0.5, 0.0, 0.0])
        np.testing.assert_allclose(float(h(pt)), -0.5, rtol=1e-12)

    def test_add_incompatible_domain_raises(self):
        F = _Function3D()
        dom1 = _make_domain()
        dom2 = _make_domain()
        f = F(dom1, evaluate_callable=lambda x: 1.0)
        g = F(dom2, evaluate_callable=lambda x: 1.0)
        with pytest.raises((ValueError, TypeError)):
            _ = f + g  # different domains

    def test_pointwise_multiplication(self):
        F = _Function3D()
        dom = _make_domain()
        f = F(dom, evaluate_callable=_x0)
        g = F(dom, evaluate_callable=_x0)
        h = f.pointwise_mul(g)
        pt = np.array([0.5, 0.0, 0.0])
        np.testing.assert_allclose(float(h(pt)), 0.25, rtol=1e-12)

    def test_pointwise_mul_returns_function3d(self):
        F = _Function3D()
        dom = _make_domain()
        f = F(dom, evaluate_callable=lambda x: 1.0)
        g = F(dom, evaluate_callable=lambda x: 2.0)
        h = f.pointwise_mul(g)
        assert isinstance(h, F)

    def test_scalar_mul_on_zero(self):
        F = _Function3D()
        dom = _make_domain()
        f = F(dom, evaluate_callable=lambda x: 0.0)
        h = 5.0 * f
        pt = np.array([0.1, 0.2, 0.3])
        np.testing.assert_allclose(float(h(pt)), 0.0, atol=1e-15)


# ===========================================================================
# copy()
# ===========================================================================

class TestFunction3DCopy:
    def test_copy_returns_new_object(self):
        F = _Function3D()
        dom = _make_domain()
        f = F(dom, evaluate_callable=lambda x: 1.0)
        fc = f.copy()
        assert fc is not f

    def test_copy_preserves_evaluation(self):
        F = _Function3D()
        dom = _make_domain()
        f = F(dom, evaluate_callable=_x0)
        fc = f.copy()
        pt = np.array([0.3, 0.0, 0.0])
        np.testing.assert_allclose(float(fc(pt)), 0.3, rtol=1e-12)

    def test_copy_preserves_domain(self):
        F = _Function3D()
        dom = _make_domain()
        f = F(dom, evaluate_callable=lambda x: 0.0)
        fc = f.copy()
        assert fc.domain is dom


# ===========================================================================
# attach_to_space / detach
# ===========================================================================

class TestFunction3DAttachDetach:
    """
    In Phase 2, without a concrete finite-basis space, we test the
    attach/detach semantics using Lebesgue3D as the target space.
    """

    def test_attach_to_space_sets_is_attached(self):
        from pygeoinf3d.core.functions import Function3D
        from pygeoinf3d.spaces.lebesgue import Lebesgue3D
        dom = _make_domain()
        f = Function3D(dom, evaluate_callable=lambda x: 1.0)
        space = Lebesgue3D(dom)
        attached = f.attach_to_space(space)
        assert attached.is_attached

    def test_attach_to_space_preserves_evaluation(self):
        from pygeoinf3d.core.functions import Function3D
        from pygeoinf3d.spaces.lebesgue import Lebesgue3D
        dom = _make_domain()
        f = Function3D(dom, evaluate_callable=_x0)
        space = Lebesgue3D(dom)
        attached = f.attach_to_space(space)
        pt = np.array([0.4, 0.0, 0.0])
        np.testing.assert_allclose(float(attached(pt)), 0.4, rtol=1e-12)

    def test_attach_to_space_has_space(self):
        from pygeoinf3d.core.functions import Function3D
        from pygeoinf3d.spaces.lebesgue import Lebesgue3D
        dom = _make_domain()
        f = Function3D(dom, evaluate_callable=lambda x: 1.0)
        space = Lebesgue3D(dom)
        attached = f.attach_to_space(space)
        assert attached.space is space

    def test_detach_returns_standalone(self):
        from pygeoinf3d.core.functions import Function3D
        from pygeoinf3d.spaces.lebesgue import Lebesgue3D
        dom = _make_domain()
        f = Function3D(dom, evaluate_callable=lambda x: 1.0)
        space = Lebesgue3D(dom)
        attached = f.attach_to_space(space)
        standalone = attached.detach()
        assert not standalone.is_attached

    def test_detach_callable_backed_preserves_eval(self):
        from pygeoinf3d.core.functions import Function3D
        from pygeoinf3d.spaces.lebesgue import Lebesgue3D
        dom = _make_domain()
        f = Function3D(dom, evaluate_callable=_x0)
        space = Lebesgue3D(dom)
        attached = f.attach_to_space(space)
        standalone = attached.detach()
        pt = np.array([0.7, 0.0, 0.0])
        np.testing.assert_allclose(float(standalone(pt)), 0.7, rtol=1e-12)

    def test_standalone_detach_raises_or_returns_self(self):
        """Detaching an already-detached function should
        either raise or return itself."""
        from pygeoinf3d.core.functions import Function3D
        dom = _make_domain()
        f = Function3D(dom, evaluate_callable=lambda x: 1.0)
        # Acceptable behaviors: raises, or returns itself/copy
        try:
            result = f.detach()
            # If it doesn't raise, result must be equivalent (not crash)
            assert not result.is_attached
        except (ValueError, RuntimeError):
            pass  # Explicitly failing on already-detached is also fine

    def test_attach_wrong_domain_raises(self):
        """Attaching to a space with a different domain should raise."""
        from pygeoinf3d.core.functions import Function3D
        from pygeoinf3d.spaces.lebesgue import Lebesgue3D
        dom1 = _make_domain()
        dom2 = _make_domain()
        f = Function3D(dom1, evaluate_callable=lambda x: 1.0)
        space2 = Lebesgue3D(dom2)
        with pytest.raises((ValueError, TypeError)):
            f.attach_to_space(space2)
