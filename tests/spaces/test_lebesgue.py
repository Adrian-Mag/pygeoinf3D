"""Tests for pygeoinf3d.spaces.lebesgue.Lebesgue3D.

Covers:
- HilbertSpace subclass contract
- Pairing / inner product consistency on a deterministic mock domain
- to_dual returns a LinearForm-compatible callable
- Evaluation of to_dual(f)(g) matches ∫ f*g
- from_dual(to_dual(f)) roundtrip by evaluation
- from_dual rejects a generic non-kernel linear form
- zero element has correct evaluation
- Space equality
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


def _x2(x):
    return x[:, 2] if x.ndim == 2 else float(x[2])


def _const1(x):
    return np.ones(len(x)) if x.ndim == 2 else 1.0


# ---------------------------------------------------------------------------
# Deterministic mock domain: unit box [0,1]^3 with analytic integration
# ---------------------------------------------------------------------------

class _MockBoundary(Boundary3D):
    def contains(self, point: np.ndarray) -> bool:
        return False


class _UnitCubeDomain(Domain3D):
    """
    Unit cube [0, 1]^3 with deterministic Monte Carlo integration.

    Uses a fixed seed and enough samples so that simple polynomial
    integrals are accurate to ~1% for Phase 2 tests.
    """

    _N_INTEGRATE = 100_000
    _SEED = 42

    def sample_interior(self, n: int, *, rng: Any = None) -> np.ndarray:
        rng = np.random.default_rng(rng)
        return rng.uniform(0.0, 1.0, size=(n, 3))

    def integrate_volume(self, f, *, method=None, **kwargs) -> float:
        n = kwargs.get("n", self._N_INTEGRATE)
        seed = kwargs.get("seed", self._SEED)
        pts = self.sample_interior(n, rng=seed)
        vals = f(pts)
        # volume of unit cube = 1
        return float(np.mean(vals))

    @property
    def boundary(self) -> Boundary3D:
        return _MockBoundary(self)


def _make_unit_cube_region() -> Region3D:
    space = EuclideanSpace(3)
    ball = Ball(space, np.array([0.5, 0.5, 0.5]), 0.5)
    bbox = AxisAlignedBoundingBox(low=np.zeros(3), high=np.ones(3))
    return Region3D(ball, bounding_box=bbox)


def _make_cube_domain() -> _UnitCubeDomain:
    return _UnitCubeDomain(_make_unit_cube_region())


# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------

def _get_classes():
    from pygeoinf3d.spaces.lebesgue import Lebesgue3D
    from pygeoinf3d.core.functions import Function3D
    return Lebesgue3D, Function3D


def _make_space_and_fns():
    """Return (space, f=x₀, g=x₁) on the unit cube."""
    Lebesgue3D, Function3D = _get_classes()
    dom = _make_cube_domain()
    space = Lebesgue3D(dom)
    # f(x) = x₀,  g(x) = x₁  — analytic ∫ f*g = ∫₀¹∫₀¹∫₀¹ x₀x₁ dx = 0.25
    f = Function3D(dom, evaluate_callable=_x0)
    g = Function3D(dom, evaluate_callable=_x1)
    return space, f, g


# ===========================================================================
# HilbertSpace contract
# ===========================================================================

class TestLebesgue3DContract:
    def test_is_hilbert_space(self):
        from pygeoinf import HilbertSpace
        Lebesgue3D, _ = _get_classes()
        dom = _make_cube_domain()
        space = Lebesgue3D(dom)
        assert isinstance(space, HilbertSpace)

    def test_dim_is_zero_basis_free(self):
        """Basis-free mode: dim == 0 signals no finite decomposition."""
        Lebesgue3D, _ = _get_classes()
        dom = _make_cube_domain()
        space = Lebesgue3D(dom)
        assert space.dim == 0

    def test_function_domain_attribute(self):
        Lebesgue3D, _ = _get_classes()
        dom = _make_cube_domain()
        space = Lebesgue3D(dom)
        assert space.function_domain is dom

    def test_equality_same_domain(self):
        Lebesgue3D, _ = _get_classes()
        dom = _make_cube_domain()
        s1 = Lebesgue3D(dom)
        s2 = Lebesgue3D(dom)
        assert s1 == s2

    def test_equality_different_domain(self):
        Lebesgue3D, _ = _get_classes()
        dom1 = _make_cube_domain()
        dom2 = _make_cube_domain()
        s1 = Lebesgue3D(dom1)
        s2 = Lebesgue3D(dom2)
        # Different domain objects → not equal
        assert s1 != s2

    def test_zero_element_evaluates_to_zero(self):
        Lebesgue3D, _ = _get_classes()
        dom = _make_cube_domain()
        space = Lebesgue3D(dom)
        z = space.zero
        pts = np.array([[0.1, 0.2, 0.3], [0.5, 0.5, 0.5]])
        np.testing.assert_allclose(z(pts), np.zeros(2), atol=1e-15)

    def test_to_components_raises_not_implemented(self):
        Lebesgue3D, Function3D = _get_classes()
        dom = _make_cube_domain()
        space = Lebesgue3D(dom)
        f = Function3D(dom, evaluate_callable=lambda x: 1.0)
        with pytest.raises(NotImplementedError):
            space.to_components(f)

    def test_from_components_raises_not_implemented(self):
        Lebesgue3D, _ = _get_classes()
        dom = _make_cube_domain()
        space = Lebesgue3D(dom)
        with pytest.raises(NotImplementedError):
            space.from_components(np.array([]))


# ===========================================================================
# to_dual / pairing
# ===========================================================================

class TestLebesgue3DToDual:
    def test_to_dual_returns_callable(self):
        space, f, g = _make_space_and_fns()
        fp = space.to_dual(f)
        assert callable(fp)

    def test_to_dual_evaluation_matches_integral(self):
        """
        to_dual(f)(g) should equal ∫ f*g over the domain.

        For f=x₀, g=x₁ on [0,1]³:  ∫ x₀·x₁ dV = (1/2)·(1/2)·1 = 1/4.
        Monte Carlo with 100k samples → tolerance 2%.
        """
        space, f, g = _make_space_and_fns()
        fp = space.to_dual(f)
        result = fp(g)
        np.testing.assert_allclose(result, 0.25, rtol=0.02)

    def test_to_dual_is_linear_form_compatible(self):
        """to_dual(f) should have a .domain attribute pointing to the space."""
        space, f, g = _make_space_and_fns()
        fp = space.to_dual(f)
        assert hasattr(fp, "domain")
        assert fp.domain is space

    def test_inner_product_via_duality(self):
        """
        inner_product(f, g) = duality_product(to_dual(f), g) = to_dual(f)(g).
        So inner_product should agree with Monte Carlo ∫ f*g.
        """
        space, f, g = _make_space_and_fns()
        ip = space.inner_product(f, g)
        np.testing.assert_allclose(ip, 0.25, rtol=0.02)

    def test_inner_product_symmetry(self):
        """⟨f, g⟩ ≈ ⟨g, f⟩ for L² on the cube."""
        space, f, g = _make_space_and_fns()
        ip_fg = space.inner_product(f, g)
        ip_gf = space.inner_product(g, f)
        np.testing.assert_allclose(ip_fg, ip_gf, rtol=0.02)

    def test_inner_product_self_nonneg(self):
        """⟨f, f⟩ >= 0."""
        space, f, _ = _make_space_and_fns()
        ip = space.inner_product(f, f)
        assert ip >= 0.0

    def test_inner_product_self_matches_mc(self):
        """
        ∫₀¹ x₀² dV = ∫₀¹ x₀² dx₀ · ∫₀¹ dx₁ · ∫₀¹ dx₂ = 1/3.
        """
        space, f, _ = _make_space_and_fns()
        ip = space.inner_product(f, f)
        np.testing.assert_allclose(ip, 1.0 / 3.0, rtol=0.02)

    def test_to_dual_linearity_scalar(self):
        """to_dual(c*f)(g) ≈ c * to_dual(f)(g)."""
        Lebesgue3D, Function3D = _get_classes()
        dom = _make_cube_domain()
        space = Lebesgue3D(dom)
        f = Function3D(dom, evaluate_callable=_x0)
        g = Function3D(dom, evaluate_callable=_x1)
        c = 3.0
        cf = c * f
        fp = space.to_dual(f)
        cfp = space.to_dual(cf)
        np.testing.assert_allclose(cfp(g), c * fp(g), rtol=0.02)


# ===========================================================================
# from_dual roundtrip
# ===========================================================================

class TestLebesgue3DFromDual:
    def test_roundtrip_by_evaluation(self):
        """
        from_dual(to_dual(f)) should return a Function3D that evaluates like f.
        """
        Lebesgue3D, Function3D = _get_classes()
        dom = _make_cube_domain()
        space = Lebesgue3D(dom)
        f = Function3D(dom, evaluate_callable=_x0)
        fp = space.to_dual(f)
        f_rec = space.from_dual(fp)
        pt = np.array([0.4, 0.2, 0.1])
        np.testing.assert_allclose(float(f_rec(pt)), 0.4, rtol=1e-10)

    def test_from_dual_returns_function3d(self):
        Lebesgue3D, Function3D = _get_classes()
        dom = _make_cube_domain()
        space = Lebesgue3D(dom)
        f = Function3D(dom, evaluate_callable=lambda x: 1.0)
        fp = space.to_dual(f)
        f_rec = space.from_dual(fp)
        assert isinstance(f_rec, Function3D)

    def test_from_dual_rejects_generic_linear_form(self):
        """
        from_dual must refuse a generic (non-kernel) LinearForm that didn't
        come from to_dual on this space.
        """
        from pygeoinf import LinearForm
        Lebesgue3D, Function3D = _get_classes()
        dom = _make_cube_domain()
        space = Lebesgue3D(dom)
        # Create a dummy EuclideanSpace and a generic LinearForm on it
        # (wrong domain) — should be rejected
        from pygeoinf.hilbert_space import EuclideanSpace as ES
        dummy_space = ES(3)
        dummy_form = LinearForm(dummy_space, components=np.ones(3))
        with pytest.raises((TypeError, ValueError)):
            space.from_dual(dummy_form)

    def test_from_dual_rejects_foreign_volume_kernel(self):
        """
        A VolumeKernel3D built on a *different* Lebesgue3D should be rejected.
        """
        Lebesgue3D, Function3D = _get_classes()
        dom1 = _make_cube_domain()
        dom2 = _make_cube_domain()
        space1 = Lebesgue3D(dom1)
        space2 = Lebesgue3D(dom2)
        f = Function3D(dom2, evaluate_callable=lambda x: 1.0)
        fp_other = space2.to_dual(f)
        with pytest.raises((TypeError, ValueError)):
            space1.from_dual(fp_other)


# ===========================================================================
# Space vector-space operations
# ===========================================================================

class TestLebesgue3DVectorOps:
    def test_zero_norm_is_zero(self):
        Lebesgue3D, Function3D = _get_classes()
        dom = _make_cube_domain()
        space = Lebesgue3D(dom)
        z = space.zero
        norm = space.norm(z)
        np.testing.assert_allclose(norm, 0.0, atol=1e-10)

    def test_norm_constant_one(self):
        """
        ||1||² = ∫₀¹∫₀¹∫₀¹ 1 dV = 1  →  ||1|| = 1.
        """
        Lebesgue3D, Function3D = _get_classes()
        dom = _make_cube_domain()
        space = Lebesgue3D(dom)
        one = Function3D(dom, evaluate_callable=_const1)
        norm = space.norm(one)
        np.testing.assert_allclose(norm, 1.0, rtol=0.02)

    def test_add_via_space(self):
        Lebesgue3D, Function3D = _get_classes()
        dom = _make_cube_domain()
        space = Lebesgue3D(dom)
        f = Function3D(dom, evaluate_callable=_x0)
        g = Function3D(dom, evaluate_callable=_x1)
        h = space.add(f, g)
        pt = np.array([0.3, 0.4, 0.0])
        np.testing.assert_allclose(float(h(pt)), 0.7, rtol=1e-12)

    def test_multiply_via_space(self):
        Lebesgue3D, Function3D = _get_classes()
        dom = _make_cube_domain()
        space = Lebesgue3D(dom)
        f = Function3D(dom, evaluate_callable=_x0)
        h = space.multiply(4.0, f)
        pt = np.array([0.5, 0.0, 0.0])
        np.testing.assert_allclose(float(h(pt)), 2.0, rtol=1e-12)


# ===========================================================================
# VolumeKernel3D linear-form algebra  (issue: inherited LinearForm arithmetic
# must not silently produce broken empty-component objects)
# ===========================================================================

class TestVolumeKernel3DAlgebra:
    """
    VolumeKernel3D overrides LinearForm arithmetic to stay kernel-backed.

    All operations must:
    - return a VolumeKernel3D (not a plain LinearForm with empty components)
    - evaluate correctly on test functions via the integration path
    """

    def _setup(self):
        from pygeoinf3d.spaces.lebesgue import Lebesgue3D
        from pygeoinf3d.core.functions import Function3D
        from pygeoinf3d.spaces.forms import VolumeKernel3D
        dom = _make_cube_domain()
        space = Lebesgue3D(dom)
        # f(x) = x₀ — kernel for testing
        f = Function3D(dom, evaluate_callable=_x0)
        # h(x) = x₁ — second kernel
        h = Function3D(dom, evaluate_callable=_x1)
        # probe g(x) = x₂
        g = Function3D(dom, evaluate_callable=_x2)
        kf = space.to_dual(f)
        kh = space.to_dual(h)
        return space, dom, f, h, g, kf, kh, VolumeKernel3D

    def test_copy_returns_volume_kernel(self):
        _, _, _, _, _, kf, _, VolumeKernel3D = self._setup()
        kf_copy = kf.copy()
        assert isinstance(kf_copy, VolumeKernel3D), (
            f"copy() returned {type(kf_copy).__name__!r}, "
            "expected VolumeKernel3D"
        )

    def test_copy_is_not_same_object(self):
        _, _, _, _, _, kf, _, _ = self._setup()
        assert kf.copy() is not kf

    def test_copy_evaluates_correctly(self):
        """copy(φ_f)(g) ≈ φ_f(g) = ∫ f·g."""
        space, dom, f, _, g, kf, _, _ = self._setup()
        kf_copy = kf.copy()
        # ∫₀¹∫₀¹∫₀¹ x₀·x₂ dV = 1/4
        np.testing.assert_allclose(kf_copy(g), 0.25, rtol=0.02)

    def test_neg_returns_volume_kernel(self):
        _, _, _, _, _, kf, _, VolumeKernel3D = self._setup()
        neg_kf = -kf
        assert isinstance(neg_kf, VolumeKernel3D), (
            f"-kf returned {type(neg_kf).__name__!r}, expected VolumeKernel3D"
        )

    def test_neg_evaluates_correctly(self):
        """(-φ_f)(g) = -φ_f(g) = -∫ f·g."""
        space, dom, f, _, g, kf, _, _ = self._setup()
        neg_kf = -kf
        np.testing.assert_allclose(neg_kf(g), -0.25, rtol=0.02)

    def test_scalar_mul_returns_volume_kernel(self):
        _, _, _, _, _, kf, _, VolumeKernel3D = self._setup()
        scaled = 3.0 * kf
        assert isinstance(scaled, VolumeKernel3D), (
            f"3.0 * kf returned {type(scaled).__name__!r}, "
            "expected VolumeKernel3D"
        )

    def test_scalar_rmul_returns_volume_kernel(self):
        _, _, _, _, _, kf, _, VolumeKernel3D = self._setup()
        scaled = kf * 3.0
        assert isinstance(scaled, VolumeKernel3D)

    def test_scalar_mul_evaluates_correctly(self):
        """(c · φ_f)(g) = c · ∫ f·g."""
        space, dom, f, _, g, kf, _, _ = self._setup()
        scaled = 3.0 * kf
        np.testing.assert_allclose(scaled(g), 3.0 * 0.25, rtol=0.02)

    def test_add_two_kernels_returns_volume_kernel(self):
        _, _, _, _, _, kf, kh, VolumeKernel3D = self._setup()
        result = kf + kh
        assert isinstance(result, VolumeKernel3D), (
            f"kf + kh returned {type(result).__name__!r}, "
            "expected VolumeKernel3D"
        )

    def test_add_evaluates_correctly(self):
        """(φ_f + φ_h)(g) = ∫(f+h)·g = ∫ x₀x₂ + x₁x₂ = 0.25 + 0.25 = 0.5."""
        space, dom, f, h, g, kf, kh, _ = self._setup()
        result = kf + kh
        np.testing.assert_allclose(result(g), 0.5, rtol=0.02)

    def test_sub_two_kernels_returns_volume_kernel(self):
        _, _, _, _, _, kf, kh, VolumeKernel3D = self._setup()
        result = kf - kh
        assert isinstance(result, VolumeKernel3D)

    def test_sub_evaluates_correctly(self):
        """(φ_f - φ_h)(g) = ∫(f-h)·g = 0.25 - 0.25 = 0."""
        space, dom, f, h, g, kf, kh, _ = self._setup()
        result = kf - kh
        np.testing.assert_allclose(result(g), 0.0, atol=0.02)

    def test_add_foreign_kernel_raises(self):
        """Adding a VolumeKernel3D from a different space must raise."""
        from pygeoinf3d.spaces.lebesgue import Lebesgue3D
        from pygeoinf3d.core.functions import Function3D
        dom1 = _make_cube_domain()
        dom2 = _make_cube_domain()
        space1 = Lebesgue3D(dom1)
        space2 = Lebesgue3D(dom2)
        f1 = Function3D(dom1, evaluate_callable=lambda x: 1.0)
        f2 = Function3D(dom2, evaluate_callable=lambda x: 1.0)
        kf1 = space1.to_dual(f1)
        kf2 = space2.to_dual(f2)
        with pytest.raises((ValueError, TypeError)):
            _ = kf1 + kf2

    def test_add_plain_linear_form_raises(self):
        """Adding a non-VolumeKernel3D LinearForm must raise."""
        from pygeoinf import LinearForm
        from pygeoinf.hilbert_space import EuclideanSpace
        from pygeoinf3d.spaces.lebesgue import Lebesgue3D
        from pygeoinf3d.core.functions import Function3D
        dom = _make_cube_domain()
        space = Lebesgue3D(dom)
        f = Function3D(dom, evaluate_callable=lambda x: 1.0)
        kf = space.to_dual(f)
        # Confirm same-space addition still works (not a raise case)
        assert kf + kf is not None
        # Adding a plain non-VolumeKernel3D LinearForm must raise
        dummy_space = EuclideanSpace(3)
        dummy_lf = LinearForm(dummy_space, components=np.ones(3))
        with pytest.raises((TypeError, ValueError)):
            _ = kf + dummy_lf

    # ------------------------------------------------------------------
    # In-place arithmetic — must override inherited LinearForm behaviour
    # which mutates the fake empty _components array silently.
    # ------------------------------------------------------------------

    def test_imul_returns_same_object(self):
        """kf *= c should return the same object (true in-place)."""
        _, _, _, _, _, kf, _, _ = self._setup()
        original_id = id(kf)
        kf *= 3.0
        assert id(kf) == original_id

    def test_imul_evaluates_correctly(self):
        """After kf *= c, kf(g) == c * original_kf(g)."""
        space, dom, f, _, g, kf, _, _ = self._setup()
        expected = 3.0 * kf(g)   # evaluate before mutation
        kf *= 3.0
        np.testing.assert_allclose(kf(g), expected, rtol=0.02)

    def test_imul_zero_zeros_form(self):
        """kf *= 0.0 should produce a zero form."""
        space, dom, f, _, g, kf, _, _ = self._setup()
        kf *= 0.0
        np.testing.assert_allclose(kf(g), 0.0, atol=1e-12)

    def test_iadd_returns_same_object(self):
        """kf += kh should return the same object (true in-place)."""
        _, _, _, _, _, kf, kh, _ = self._setup()
        original_id = id(kf)
        kf += kh
        assert id(kf) == original_id

    def test_iadd_evaluates_correctly(self):
        """After kf += kh, kf(g) == original_kf(g) + kh(g)."""
        space, dom, f, h, g, kf, kh, _ = self._setup()
        before = kf(g) + kh(g)   # ≈ 0.25 + 0.25 = 0.50
        kf += kh
        np.testing.assert_allclose(kf(g), before, rtol=0.02)

    def test_iadd_foreign_raises(self):
        """kf += kernel_from_different_space must raise ValueError."""
        from pygeoinf3d.spaces.lebesgue import Lebesgue3D
        from pygeoinf3d.core.functions import Function3D
        dom1 = _make_cube_domain()
        dom2 = _make_cube_domain()
        space1 = Lebesgue3D(dom1)
        space2 = Lebesgue3D(dom2)
        f1 = Function3D(dom1, evaluate_callable=lambda x: 1.0)
        f2 = Function3D(dom2, evaluate_callable=lambda x: 1.0)
        kf1 = space1.to_dual(f1)
        kf2 = space2.to_dual(f2)
        with pytest.raises((ValueError, TypeError)):
            kf1 += kf2

    def test_iadd_plain_linear_form_raises(self):
        """kf += plain LinearForm must raise TypeError."""
        from pygeoinf import LinearForm
        from pygeoinf.hilbert_space import EuclideanSpace
        from pygeoinf3d.spaces.lebesgue import Lebesgue3D
        from pygeoinf3d.core.functions import Function3D
        dom = _make_cube_domain()
        space = Lebesgue3D(dom)
        f = Function3D(dom, evaluate_callable=lambda x: 1.0)
        kf = space.to_dual(f)
        dummy_lf = LinearForm(EuclideanSpace(3), components=np.ones(3))
        with pytest.raises(TypeError):
            kf += dummy_lf

    def test_isub_returns_same_object(self):
        """kf -= kh should return the same object (true in-place)."""
        _, _, _, _, _, kf, kh, _ = self._setup()
        original_id = id(kf)
        kf -= kh
        assert id(kf) == original_id

    def test_isub_evaluates_correctly(self):
        """After kf -= kh, kf(g) == original_kf(g) - kh(g)."""
        space, dom, f, h, g, kf, kh, _ = self._setup()
        before = kf(g) - kh(g)   # ≈ 0.0
        kf -= kh
        np.testing.assert_allclose(kf(g), before, atol=0.02)

    def test_isub_foreign_raises(self):
        """kf -= kernel_from_different_space must raise ValueError."""
        from pygeoinf3d.spaces.lebesgue import Lebesgue3D
        from pygeoinf3d.core.functions import Function3D
        dom1 = _make_cube_domain()
        dom2 = _make_cube_domain()
        space1 = Lebesgue3D(dom1)
        space2 = Lebesgue3D(dom2)
        f1 = Function3D(dom1, evaluate_callable=lambda x: 1.0)
        f2 = Function3D(dom2, evaluate_callable=lambda x: 1.0)
        kf1 = space1.to_dual(f1)
        kf2 = space2.to_dual(f2)
        with pytest.raises((ValueError, TypeError)):
            kf1 -= kf2


# ===========================================================================
# Domain-identity guards (Phase 2 revision)
# ===========================================================================

class TestDomainIdentityGuards:
    """Guards that prevent silent misuse when domains do not match."""

    # ------------------------------------------------------------------
    # to_dual rejects wrong-domain functions
    # ------------------------------------------------------------------

    def test_to_dual_rejects_function_on_wrong_domain(self):
        """to_dual must raise ValueError
        when f.domain != space.function_domain."""
        from pygeoinf3d.spaces.lebesgue import Lebesgue3D
        from pygeoinf3d.core.functions import Function3D
        dom1 = _make_cube_domain()
        dom2 = _make_cube_domain()
        space = Lebesgue3D(dom1)
        # f lives on dom2, not dom1
        f_wrong = Function3D(dom2, evaluate_callable=lambda x: 1.0)
        with pytest.raises(ValueError, match="identity check"):
            space.to_dual(f_wrong)

    def test_to_dual_accepts_correct_domain(self):
        """to_dual must NOT raise when f.domain
        is exactly space.function_domain."""
        from pygeoinf3d.spaces.lebesgue import Lebesgue3D
        from pygeoinf3d.core.functions import Function3D
        dom = _make_cube_domain()
        space = Lebesgue3D(dom)
        f = Function3D(dom, evaluate_callable=lambda x: 1.0)
        # Should not raise
        kf = space.to_dual(f)
        assert kf is not None

    # ------------------------------------------------------------------
    # VolumeKernel3D construction guard
    # ------------------------------------------------------------------

    def test_volume_kernel_rejects_wrong_domain(self):
        """VolumeKernel3D.__init__ must raise ValueError
        for domain mismatch."""
        from pygeoinf3d.spaces.lebesgue import Lebesgue3D
        from pygeoinf3d.core.functions import Function3D
        from pygeoinf3d.spaces.forms import VolumeKernel3D
        dom1 = _make_cube_domain()
        dom2 = _make_cube_domain()
        space = Lebesgue3D(dom1)
        f_wrong = Function3D(dom2, evaluate_callable=lambda x: 1.0)
        with pytest.raises(ValueError, match="identity check"):
            VolumeKernel3D(space, f_wrong)

    # ------------------------------------------------------------------
    # components property is blocked
    # ------------------------------------------------------------------

    def test_components_property_raises(self):
        """VolumeKernel3D.components must raise
        TypeError in basis-free mode."""
        from pygeoinf3d.spaces.lebesgue import Lebesgue3D
        from pygeoinf3d.core.functions import Function3D
        dom = _make_cube_domain()
        space = Lebesgue3D(dom)
        f = Function3D(dom, evaluate_callable=lambda x: 1.0)
        kf = space.to_dual(f)
        with pytest.raises(
            TypeError, match="no finite-component representation"
        ):
            _ = kf.components

    def test_plain_linear_form_on_left_raises(self):
        """plain_lf + volume_kernel must not silently
        produce a broken object."""
        from pygeoinf import LinearForm
        from pygeoinf3d.spaces.lebesgue import Lebesgue3D
        from pygeoinf3d.core.functions import Function3D
        # Build a LinearForm on a small EuclideanSpace — wrong type entirely
        from pygeoinf.hilbert_space import EuclideanSpace
        dummy_space = EuclideanSpace(3)
        plain_lf = LinearForm(dummy_space, components=np.ones(3))

        dom = _make_cube_domain()
        space = Lebesgue3D(dom)
        f = Function3D(dom, evaluate_callable=lambda x: 1.0)
        kf = space.to_dual(f)

        # "plain_lf + kf" should raise — not silently produce garbage.
        # The error may come from LinearForm.__add__
        # attempting other.components.
        with pytest.raises((TypeError, ValueError)):
            _ = plain_lf + kf
