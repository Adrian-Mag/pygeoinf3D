"""Tests for pygeoinf3d.spaces.sobolev.Sobolev3D.

Covers (Phase 1):
- Constructor stores base_space, weight, inverse_weight
- Constructor validates type and domain compatibility
- to_dual returns VolumeKernel3D on base_space with kernel W*f
- from_dual round-trip: from_dual(to_dual(f)) ≈ f pointwise
- from_dual rejects wrong type and wrong-domain VolumeKernel3D
- inner_product agrees with ⟨Wf, g⟩_{L²}
- norm² = inner_product(f, f)
- Equality / inequality
- Compatibility with BesselSobolevBox on BoxDomain
- zero element evaluates to 0
"""

from __future__ import annotations

import pytest
import numpy as np
import numpy.testing as npt
from typing import Any

from pygeoinf import EuclideanSpace, Ball
from pygeoinf.linear_operators import LinearOperator

from pygeoinf3d.core.config import AxisAlignedBoundingBox
from pygeoinf3d.core.region import Region3D
from pygeoinf3d.core.domain_base import Domain3D
from pygeoinf3d.core.boundary import Boundary3D
from pygeoinf3d.core.functions import Function3D
from pygeoinf3d.spaces.lebesgue import Lebesgue3D
from pygeoinf3d.spaces.forms import VolumeKernel3D
from pygeoinf3d.spaces.sobolev import Sobolev3D


# ---------------------------------------------------------------------------
# Minimal mock domain — unit cube [0, 1]^3 with deterministic MC integration
# ---------------------------------------------------------------------------

class _MockBoundary(Boundary3D):
    def contains(self, point: np.ndarray) -> bool:
        return False


class _UnitCubeDomain(Domain3D):
    """Unit cube [0,1]^3 with MC integration (fixed seed, large n)."""

    _N = 200_000
    _SEED = 42

    def sample_interior(self, n: int, *, rng: Any = None) -> np.ndarray:
        rng = np.random.default_rng(rng)
        return rng.uniform(0.0, 1.0, size=(n, 3))

    def integrate_volume(self, f, *, method=None, **kwargs) -> float:
        n = kwargs.get("n", self._N)
        seed = kwargs.get("seed", self._SEED)
        pts = self.sample_interior(n, rng=seed)
        return float(np.mean(f(pts)))

    @property
    def boundary(self) -> Boundary3D:
        return _MockBoundary(self)

    def structured_mesh(self, resolution, *, location="cell_centers"):
        raise NotImplementedError

    def volume_weights(self, resolution, **kwargs):
        raise NotImplementedError

    @property
    def coordinate_system(self):
        raise NotImplementedError

    @property
    def boundary_components(self):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Helper: build a scalar-multiple weight operator
# ---------------------------------------------------------------------------

def _make_alpha_weight(space: Lebesgue3D, alpha: float) -> LinearOperator:
    """Return a LinearOperator on *space* that scales every f(x) by *alpha*.

    The operator is self-adjoint in L².  The inverse is the same operator
    with coefficient ``1/alpha``.
    """
    def apply(f: Function3D) -> Function3D:
        a = float(alpha)
        return Function3D(
            f.domain,
            evaluate_callable=lambda x: a * f(x),
        )

    return LinearOperator(space, space, apply, adjoint_mapping=apply)


# ---------------------------------------------------------------------------
# Module-level callables (avoids lambda capture issues)
# ---------------------------------------------------------------------------

def _f_x0(x: np.ndarray):
    """f(x) = x₀."""
    return x[:, 0] if x.ndim == 2 else float(x[0])


def _g_const1(x: np.ndarray):
    """g(x) = 1."""
    return np.ones(len(x)) if x.ndim == 2 else 1.0


def _h_x1sq(x: np.ndarray):
    """h(x) = x₁²."""
    return (x[:, 1] ** 2) if x.ndim == 2 else float(x[1] ** 2)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def domain():
    return _UnitCubeDomain(
        Region3D(
            Ball(EuclideanSpace(3), np.zeros(3), 1.0),
            bounding_box=AxisAlignedBoundingBox(np.zeros(3), np.ones(3)),
        )
    )


@pytest.fixture(scope="module")
def base_space(domain):
    return Lebesgue3D(domain)


@pytest.fixture(scope="module")
def weight_alpha2(base_space):
    """Weight operator W = 2·I."""
    return _make_alpha_weight(base_space, 2.0)


@pytest.fixture(scope="module")
def inv_weight_alpha2(base_space):
    """Inverse weight operator W⁻¹ = ½·I."""
    return _make_alpha_weight(base_space, 0.5)


@pytest.fixture(scope="module")
def sobolev(base_space, weight_alpha2, inv_weight_alpha2):
    return Sobolev3D(base_space, weight_alpha2, inv_weight_alpha2)


@pytest.fixture(scope="module")
def f_func(domain):
    return Function3D(domain, evaluate_callable=_f_x0)


@pytest.fixture(scope="module")
def g_func(domain):
    return Function3D(domain, evaluate_callable=_g_const1)


# ---------------------------------------------------------------------------
# TestSobolev3DConstructor
# ---------------------------------------------------------------------------

class TestSobolev3DConstructor:
    """Constructor stores all fields and exposes the right properties."""

    def test_construct_succeeds(self, sobolev):
        assert sobolev is not None

    def test_base_space_property(self, sobolev, base_space):
        assert sobolev.base_space is base_space

    def test_weight_property(self, sobolev, weight_alpha2):
        assert sobolev.weight is weight_alpha2

    def test_inverse_weight_property(self, sobolev, inv_weight_alpha2):
        assert sobolev.inverse_weight is inv_weight_alpha2

    def test_function_domain_property(self, sobolev, domain):
        assert sobolev.function_domain is domain

    def test_dim_is_zero_basis_free(self, sobolev):
        assert sobolev.dim == 0


# ---------------------------------------------------------------------------
# TestSobolev3DCompatibilityValidation
# ---------------------------------------------------------------------------

class TestSobolev3DCompatibilityValidation:
    """Constructor raises on incompatible weight domains."""

    def test_raises_if_base_space_not_lebesgue3d(
        self, domain, weight_alpha2, inv_weight_alpha2
    ):
        with pytest.raises(TypeError, match="Lebesgue3D"):
            Sobolev3D(domain, weight_alpha2, inv_weight_alpha2)

    def test_raises_if_weight_domain_mismatch(
        self, base_space, inv_weight_alpha2
    ):
        other_domain = _UnitCubeDomain(
            Region3D(
                Ball(EuclideanSpace(3), np.zeros(3), 1.0),
                bounding_box=AxisAlignedBoundingBox(np.zeros(3), np.ones(3)),
            )
        )
        other_space = Lebesgue3D(other_domain)
        wrong_weight = _make_alpha_weight(other_space, 2.0)
        with pytest.raises(ValueError, match="weight.domain"):
            Sobolev3D(base_space, wrong_weight, inv_weight_alpha2)

    def test_raises_if_weight_codomain_mismatch(
        self, base_space, inv_weight_alpha2
    ):
        other_domain = _UnitCubeDomain(
            Region3D(
                Ball(EuclideanSpace(3), np.zeros(3), 1.0),
                bounding_box=AxisAlignedBoundingBox(np.zeros(3), np.ones(3)),
            )
        )
        other_space = Lebesgue3D(other_domain)
        # Build a weight whose codomain is other_space (wrong)
        wrong_weight = LinearOperator(
            base_space,
            other_space,
            lambda f: Function3D(
                other_domain, evaluate_callable=lambda x: 2.0 * f(x)
            ),
        )
        with pytest.raises(ValueError, match="weight.codomain"):
            Sobolev3D(base_space, wrong_weight, inv_weight_alpha2)

    def test_raises_if_inverse_weight_domain_mismatch(
        self, base_space, weight_alpha2
    ):
        other_domain = _UnitCubeDomain(
            Region3D(
                Ball(EuclideanSpace(3), np.zeros(3), 1.0),
                bounding_box=AxisAlignedBoundingBox(np.zeros(3), np.ones(3)),
            )
        )
        other_space = Lebesgue3D(other_domain)
        wrong_inv = _make_alpha_weight(other_space, 0.5)
        with pytest.raises(ValueError, match="inverse_weight.domain"):
            Sobolev3D(base_space, weight_alpha2, wrong_inv)

    def test_raises_if_inverse_weight_codomain_mismatch(
        self, base_space, weight_alpha2
    ):
        other_domain = _UnitCubeDomain(
            Region3D(
                Ball(EuclideanSpace(3), np.zeros(3), 1.0),
                bounding_box=AxisAlignedBoundingBox(np.zeros(3), np.ones(3)),
            )
        )
        other_space = Lebesgue3D(other_domain)
        # Build an inverse_weight whose codomain is other_space (wrong)
        wrong_inv = LinearOperator(
            base_space,
            other_space,
            lambda f: Function3D(
                other_domain, evaluate_callable=lambda x: 0.5 * f(x)
            ),
        )
        with pytest.raises(ValueError, match="inverse_weight.codomain"):
            Sobolev3D(base_space, weight_alpha2, wrong_inv)

    def test_raises_if_weight_not_operator(
        self, base_space, inv_weight_alpha2
    ):
        """TypeError if weight has no .domain / .codomain attributes."""
        with pytest.raises(TypeError, match="weight"):
            Sobolev3D(base_space, "not_an_operator", inv_weight_alpha2)

    def test_raises_if_inverse_weight_not_operator(
        self, base_space, weight_alpha2
    ):
        """TypeError if inverse_weight lacks .domain / .codomain.
        """
        with pytest.raises(TypeError, match="inverse_weight"):
            Sobolev3D(base_space, weight_alpha2, 42)


# ---------------------------------------------------------------------------
# TestSobolev3DToDual
# ---------------------------------------------------------------------------

class TestSobolev3DToDual:
    """to_dual returns a VolumeKernel3D encoding g ↦ ⟨Wf, g⟩_{L²}."""

    def test_to_dual_returns_volume_kernel3d(self, sobolev, f_func):
        fp = sobolev.to_dual(f_func)
        assert isinstance(fp, VolumeKernel3D)

    def test_to_dual_kernel_domain_is_base_space(
        self, sobolev, base_space, f_func
    ):
        fp = sobolev.to_dual(f_func)
        assert fp.domain is base_space

    def test_to_dual_kernel_is_wf(self, sobolev, f_func):
        """Kernel of to_dual(f) should evaluate to W*f pointwise."""
        fp = sobolev.to_dual(f_func)
        pts = np.array([[0.1, 0.2, 0.3], [0.5, 0.5, 0.5]])
        expected = 2.0 * _f_x0(pts)   # W = 2·I
        npt.assert_allclose(fp.kernel(pts), expected, rtol=1e-10)

    def test_to_dual_evaluation_is_weighted_l2(self, sobolev, f_func, g_func):
        """fp(g) == ⟨Wf, g⟩_{L²} ≈ 2 * ∫ x₀ dV = 2 * 0.5 = 1.0."""
        fp = sobolev.to_dual(f_func)
        val = fp(g_func)
        # ∫₀¹∫₀¹∫₀¹  2·x₀ · 1  dx₀ dx₁ dx₂  = 2 * 0.5 = 1.0
        npt.assert_allclose(val, 1.0, atol=5e-3)

    def test_to_dual_equals_inner_product(self, sobolev, f_func, g_func):
        """Direct Riesz consistency.

        `to_dual(f)(g)` must equal `inner_product(f, g)`.
        """
        val_dual = sobolev.to_dual(f_func)(g_func)
        val_ip = sobolev.inner_product(f_func, g_func)
        npt.assert_allclose(val_dual, val_ip, rtol=1e-10)


# ---------------------------------------------------------------------------
# TestSobolev3DFromDual
# ---------------------------------------------------------------------------

class TestSobolev3DFromDual:
    """from_dual(to_dual(f)) round-trips and validates its input."""

    def test_roundtrip_pointwise(self, sobolev, f_func):
        """from_dual(to_dual(f)) evaluates identically to f."""
        fp = sobolev.to_dual(f_func)
        f_recovered = sobolev.from_dual(fp)
        pts = np.array([[0.1, 0.2, 0.3], [0.7, 0.0, 1.0]])
        npt.assert_allclose(f_recovered(pts), f_func(pts), rtol=1e-10)

    def test_roundtrip_quadratic(self, sobolev, domain):
        """Round-trip on h(x) = x₁²."""
        h = Function3D(domain, evaluate_callable=_h_x1sq)
        hp = sobolev.to_dual(h)
        h_rec = sobolev.from_dual(hp)
        pts = np.linspace(0.0, 1.0, 5).reshape(-1, 1) * np.ones((5, 3))
        npt.assert_allclose(h_rec(pts), h(pts), rtol=1e-10)

    def test_rejects_non_volume_kernel(self, sobolev):
        with pytest.raises(TypeError, match="VolumeKernel3D"):
            sobolev.from_dual(object())

    def test_rejects_volume_kernel_from_wrong_space(self, sobolev, f_func):
        """A VolumeKernel3D bound to a different Lebesgue3D is rejected."""
        # Build a second space on a different domain instance
        other_domain = _UnitCubeDomain(
            Region3D(
                Ball(EuclideanSpace(3), np.zeros(3), 1.0),
                bounding_box=AxisAlignedBoundingBox(np.zeros(3), np.ones(3)),
            )
        )
        other_space = Lebesgue3D(other_domain)
        other_f = Function3D(other_domain, evaluate_callable=_f_x0)
        foreign_form = VolumeKernel3D(other_space, other_f)
        with pytest.raises(ValueError, match="base_space"):
            sobolev.from_dual(foreign_form)


# ---------------------------------------------------------------------------
# TestSobolev3DInnerProduct
# ---------------------------------------------------------------------------

class TestSobolev3DInnerProduct:
    """inner_product(f, g) = ⟨Wf, g⟩_{L²} = base.inner_product(W·f, g)."""

    def test_inner_product_agrees_with_weighted_l2(
        self, sobolev, base_space, f_func, g_func, weight_alpha2
    ):
        """sobolev.inner_product(f, g) == base.inner_product(W·f, g)."""
        h_ip = sobolev.inner_product(f_func, g_func)
        Wf = weight_alpha2(f_func)
        ref_ip = base_space.inner_product(Wf, g_func)
        npt.assert_allclose(h_ip, ref_ip, rtol=1e-6)

    def test_inner_product_analytic_value(self, sobolev, f_func, g_func):
        """⟨f, 1⟩_H with W=2I = 2*∫₀¹ x₀ dV = 1.0."""
        ip = sobolev.inner_product(f_func, g_func)
        npt.assert_allclose(ip, 1.0, atol=5e-3)

    def test_inner_product_symmetric(self, sobolev, f_func, g_func):
        """⟨f, g⟩ = ⟨g, f⟩ when W is self-adjoint."""
        ip_fg = sobolev.inner_product(f_func, g_func)
        ip_gf = sobolev.inner_product(g_func, f_func)
        npt.assert_allclose(ip_fg, ip_gf, rtol=1e-6)

    def test_norm_squared_via_inner_product(self, sobolev, f_func):
        """‖f‖² via inner_product should be non-negative."""
        ip = sobolev.inner_product(f_func, f_func)
        assert ip > 0.0

    def test_norm_squared_equals_inner_product_ff(self, sobolev, f_func):
        """`norm(f)**2` matches `inner_product(f, f)` up to precision.
        """
        ip_ff = sobolev.inner_product(f_func, f_func)
        norm_f = sobolev.norm(f_func)
        npt.assert_allclose(norm_f ** 2, ip_ff, rtol=1e-12)


# ---------------------------------------------------------------------------
# TestSobolev3DZeroAndArithmetic
# ---------------------------------------------------------------------------

class TestSobolev3DZeroAndArithmetic:
    """zero property and vector-space operations."""

    def test_zero_element_evaluates_to_zero(self, sobolev):
        z = sobolev.zero
        pts = np.array([[0.3, 0.4, 0.5], [0.0, 1.0, 0.5]])
        npt.assert_allclose(z(pts), 0.0, atol=1e-14)

    def test_add_two_functions(self, sobolev, f_func, g_func):
        s = sobolev.add(f_func, g_func)
        pts = np.array([[0.2, 0.3, 0.4]])
        npt.assert_allclose(s(pts), f_func(pts) + g_func(pts), rtol=1e-10)

    def test_multiply_by_scalar(self, sobolev, f_func):
        res = sobolev.multiply(3.0, f_func)
        pts = np.array([[0.5, 0.0, 0.0]])
        npt.assert_allclose(res(pts), 3.0 * f_func(pts), rtol=1e-10)


# ---------------------------------------------------------------------------
# TestSobolev3DEquality
# ---------------------------------------------------------------------------

class TestSobolev3DEquality:
    """Equality is based on base_space, weight, and inverse_weight."""

    def test_equal_to_itself(self, sobolev):
        assert sobolev == sobolev

    def test_different_alpha_not_equal(
        self, base_space, weight_alpha2, inv_weight_alpha2
    ):
        w3 = _make_alpha_weight(base_space, 3.0)
        iw3 = _make_alpha_weight(base_space, 1.0 / 3.0)
        s2 = Sobolev3D(base_space, weight_alpha2, inv_weight_alpha2)
        s3 = Sobolev3D(base_space, w3, iw3)
        # Two different operator instances → not equal (identity-based)
        assert s2 != s3

    def test_not_equal_to_lebesgue3d(self, sobolev, base_space):
        assert sobolev != base_space

    def test_two_instances_same_operators_equal(
        self, base_space, weight_alpha2, inv_weight_alpha2
    ):
        s1 = Sobolev3D(base_space, weight_alpha2, inv_weight_alpha2)
        s2 = Sobolev3D(base_space, weight_alpha2, inv_weight_alpha2)
        # Same operator objects → equal
        assert s1 == s2


# ---------------------------------------------------------------------------
# TestSobolev3DToComponentsRaisesNotImplemented
# ---------------------------------------------------------------------------

class TestSobolev3DComponentsNotImplemented:
    """Basis-free component conversions raise NotImplementedError.
    """

    def test_to_components_raises(self, sobolev, f_func):
        with pytest.raises(NotImplementedError):
            sobolev.to_components(f_func)

    def test_from_components_raises(self, sobolev):
        with pytest.raises(NotImplementedError):
            sobolev.from_components(np.zeros(5))


# ---------------------------------------------------------------------------
# TestSobolev3DRepr
# ---------------------------------------------------------------------------

class TestSobolev3DRepr:
    def test_repr_contains_class_name(self, sobolev):
        r = repr(sobolev)
        assert "Sobolev3D" in r


# ---------------------------------------------------------------------------
# TestSobolev3DBesselSobolevBoxCompatibility
# ---------------------------------------------------------------------------

class TestSobolev3DBesselSobolevBoxCompatibility:
    """Sobolev3D can be constructed with BesselSobolevBox as weight."""

    def test_construct_with_bessel_sobolev_box(self):
        from pygeoinf3d.geometries.box.domain import BoxDomain
        from pygeoinf3d.providers.tensor_product import TensorProductProvider
        from pygeoinf3d.operators.bessel_sobolev_box import BesselSobolevBox

        box = BoxDomain(np.zeros(3), np.ones(3))
        space = Lebesgue3D(box)
        provider = TensorProductProvider(
            box, truncation=(3, 3, 3), bc_type="neumann"
        )

        w = BesselSobolevBox(space, provider, shift=1.0, order=1.0)
        w_inv = BesselSobolevBox(space, provider, shift=1.0, order=-1.0)

        sob = Sobolev3D(space, w, w_inv)
        assert sob is not None
        assert sob.base_space is space
        assert sob.weight is w
        assert sob.inverse_weight is w_inv

    def test_roundtrip_with_bessel_sobolev_box(self):
        """from_dual(to_dual(f)) ≈ f when weight = BesselSobolevBox."""
        from pygeoinf3d.geometries.box.domain import BoxDomain
        from pygeoinf3d.providers.tensor_product import TensorProductProvider
        from pygeoinf3d.operators.bessel_sobolev_box import BesselSobolevBox

        box = BoxDomain(np.zeros(3), np.ones(3))
        space = Lebesgue3D(box)
        provider = TensorProductProvider(
            box, truncation=(4, 4, 4), bc_type="neumann"
        )

        w = BesselSobolevBox(space, provider, shift=1.0, order=2.0)
        w_inv = BesselSobolevBox(space, provider, shift=1.0, order=-2.0)

        sob = Sobolev3D(space, w, w_inv)

        # f = constant 1 function (representable in Neumann basis)
        f = Function3D(
            box,
            evaluate_callable=lambda x: (
                np.ones(len(x)) if x.ndim == 2 else 1.0
            ),
        )
        fp = sob.to_dual(f)
        f_rec = sob.from_dual(fp)

        # Evaluate at a few interior points
        pts = np.array([
            [0.5, 0.5, 0.5],
            [0.25, 0.75, 0.25],
        ])
        npt.assert_allclose(f_rec(pts), f(pts), atol=0.05)

    def test_inner_product_positive_definite_with_bessel(self):
        """⟨f, f⟩_H > 0 with BesselSobolevBox weight."""
        from pygeoinf3d.geometries.box.domain import BoxDomain
        from pygeoinf3d.providers.tensor_product import TensorProductProvider
        from pygeoinf3d.operators.bessel_sobolev_box import BesselSobolevBox

        box = BoxDomain(np.zeros(3), np.ones(3))
        space = Lebesgue3D(box)
        provider = TensorProductProvider(
            box, truncation=(3, 3, 3), bc_type="neumann"
        )

        w = BesselSobolevBox(space, provider, shift=1.0, order=2.0)
        w_inv = BesselSobolevBox(space, provider, shift=1.0, order=-2.0)

        sob = Sobolev3D(space, w, w_inv)

        f = Function3D(
            box,
            evaluate_callable=lambda x: (
                x[:, 0] if x.ndim == 2 else float(x[0])
            ),
        )
        ip = sob.inner_product(f, f)
        assert ip > 0.0


# ---------------------------------------------------------------------------
# TestSobolev3DFromDualSpectral
# ---------------------------------------------------------------------------

class TestSobolev3DFromDualSpectral:
    """from_dual handles PathIntegralFunctional and SumLinearForm via
    spectral representer.
    """

    @pytest.fixture
    def sobolev_bessel(self):
        from pygeoinf3d.geometries.box.domain import BoxDomain
        from pygeoinf3d.providers.tensor_product import TensorProductProvider
        from pygeoinf3d.operators.bessel_sobolev_box import BesselSobolevBox

        box = BoxDomain(np.zeros(3), np.ones(3))
        space = Lebesgue3D(box)
        provider = TensorProductProvider(box, truncation=(5, 5, 5), bc_type="neumann")
        w = BesselSobolevBox(space, provider, shift=1.0, order=4.0)
        w_inv = BesselSobolevBox(space, provider, shift=1.0, order=-4.0)
        return Sobolev3D(space, w, w_inv)

    @pytest.fixture
    def path_across_box(self):
        from pygeoinf3d.core.paths import Path3D
        start = np.array([0.1, 0.5, 0.5])
        end = np.array([0.9, 0.5, 0.5])
        return Path3D.from_polyline(np.array([start, end]))

    @pytest.fixture
    def path_diagonal(self):
        from pygeoinf3d.core.paths import Path3D
        return Path3D.from_polyline(
            np.array([[0.1, 0.1, 0.1], [0.9, 0.9, 0.9]])
        )

    def test_from_dual_path_integral_returns_function3d(
        self, sobolev_bessel, path_across_box
    ):
        from pygeoinf3d.spaces.forms import PathIntegralFunctional

        form = PathIntegralFunctional(
            sobolev_bessel.base_space, path_across_box, n_quad=50
        )
        r = sobolev_bessel.from_dual(form)
        assert isinstance(r, Function3D)

    def test_from_dual_path_integral_representer_identity(
        self, sobolev_bessel, path_across_box
    ):
        """ℓ_γ(r_γ) ≈ ‖r_γ‖_H² (representer identity)."""
        from pygeoinf3d.spaces.forms import PathIntegralFunctional

        form = PathIntegralFunctional(
            sobolev_bessel.base_space, path_across_box, n_quad=50
        )
        r = sobolev_bessel.from_dual(form)
        lhs = float(form(r))
        rhs = sobolev_bessel.inner_product(r, r)
        npt.assert_allclose(lhs, rhs, rtol=0.10)

    def test_from_dual_path_integral_representer_theorem(
        self, sobolev_bessel, path_across_box
    ):
        """<r_γ, φ_0>_H ≈ ℓ_γ(φ_0) (representer theorem on eigenfunction)."""
        from pygeoinf3d.spaces.forms import PathIntegralFunctional

        form = PathIntegralFunctional(
            sobolev_bessel.base_space, path_across_box, n_quad=50
        )
        r = sobolev_bessel.from_dual(form)
        inv_w = sobolev_bessel.inverse_weight
        phi0 = inv_w.get_eigenfunction(0)
        lhs = sobolev_bessel.inner_product(r, phi0)
        rhs = float(form(phi0))
        npt.assert_allclose(lhs, rhs, rtol=0.01)

    def test_from_dual_sum_linear_form_returns_function3d(
        self, sobolev_bessel, path_across_box, path_diagonal
    ):
        """from_dual(SumLinearForm) returns a Function3D."""
        from pygeoinf3d.spaces.forms import PathIntegralFunctional, SumLinearForm

        f1 = PathIntegralFunctional(
            sobolev_bessel.base_space, path_across_box, n_quad=50
        )
        f2 = PathIntegralFunctional(
            sobolev_bessel.base_space, path_diagonal, n_quad=50
        )
        sum_form = SumLinearForm(sobolev_bessel.base_space, [(1.0, f1), (1.0, f2)])
        r = sobolev_bessel.from_dual(sum_form)
        assert isinstance(r, Function3D)

    def test_from_dual_sum_linear_form_representer_identity(
        self, sobolev_bessel, path_across_box, path_diagonal
    ):
        """ℓ_sum(r_sum) ≈ ‖r_sum‖_H²."""
        from pygeoinf3d.spaces.forms import PathIntegralFunctional, SumLinearForm

        f1 = PathIntegralFunctional(
            sobolev_bessel.base_space, path_across_box, n_quad=50
        )
        f2 = PathIntegralFunctional(
            sobolev_bessel.base_space, path_diagonal, n_quad=50
        )
        sum_form = SumLinearForm(
            sobolev_bessel.base_space, [(0.5, f1), (0.5, f2)]
        )
        r = sobolev_bessel.from_dual(sum_form)
        lhs = float(sum_form(r))
        rhs = sobolev_bessel.inner_product(r, r)
        npt.assert_allclose(lhs, rhs, rtol=0.15)

    def test_from_dual_non_spectral_inverse_weight_raises(self, path_across_box):
        """Raises TypeError if inverse_weight is not a SpectralOperator3D."""
        from pygeoinf3d.spaces.forms import PathIntegralFunctional
        from pygeoinf3d.geometries.box.domain import BoxDomain
        from pygeoinf.linear_operators import LinearOperator

        box = BoxDomain(np.zeros(3), np.ones(3))
        space = Lebesgue3D(box)
        W = LinearOperator(
            space, space, lambda f: f, adjoint_mapping=lambda f: f
        )
        sob = Sobolev3D(space, W, W)
        form = PathIntegralFunctional(space, path_across_box, n_quad=20)
        with pytest.raises(TypeError, match="SpectralOperator3D"):
            sob.from_dual(form)

    def test_sola_adjoint_with_sobolev_domain(
        self, sobolev_bessel, path_across_box, path_diagonal
    ):
        """SOLAOperator3D.adjoint works when domain is Sobolev3D."""
        from pygeoinf import EuclideanSpace
        from pygeoinf3d.spaces.forms import PathIntegralFunctional
        from pygeoinf3d.operators.sola import SOLAOperator3D

        f1 = PathIntegralFunctional(
            sobolev_bessel.base_space, path_across_box, n_quad=50
        )
        f2 = PathIntegralFunctional(
            sobolev_bessel.base_space, path_diagonal, n_quad=50
        )
        op = SOLAOperator3D(sobolev_bessel, EuclideanSpace(2), [f1, f2])
        y = np.array([1.0, 0.5])
        adj_y = op.adjoint(y)
        assert isinstance(adj_y, Function3D)

    def test_sola_adjoint_consistency(self, sobolev_bessel, path_across_box):
        """<G*y, f>_H ≈ <y, Gf>_{R^n} (adjoint consistency check)."""
        from pygeoinf import EuclideanSpace
        from pygeoinf3d.spaces.forms import PathIntegralFunctional
        from pygeoinf3d.operators.sola import SOLAOperator3D

        form = PathIntegralFunctional(
            sobolev_bessel.base_space, path_across_box, n_quad=50
        )
        op = SOLAOperator3D(sobolev_bessel, EuclideanSpace(1), [form])
        phi0 = sobolev_bessel.inverse_weight.get_eigenfunction(0)
        y = np.array([1.0])
        Gf = op(phi0)
        Gstar_y = op.adjoint(y)
        lhs = sobolev_bessel.inner_product(Gstar_y, phi0)
        rhs = float(np.dot(y, Gf))
        npt.assert_allclose(lhs, rhs, rtol=0.05)
