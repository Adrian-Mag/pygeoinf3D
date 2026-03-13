"""Tests for pygeoinf3d.spaces.sobolev.Sobolev3D.

Covers:
- Construction and basic properties (dim, s, kappa, eigenvalues, weights)
- Component identity: to_components / from_components
- Weighted inner product and self-adjointness
- Dual mapping: to_dual / from_dual roundtrip
- Spectral conversion: project and to_function
- Trace-readiness guards: supports_trace / assert_trace_allowed
- pygeoinf compatibility smoke tests (zero, random,
  inner_product / duality_product)
- BoxDomain sample_boundary / outward_normal trace hooks
"""

from __future__ import annotations

import pytest
import numpy as np
import numpy.testing as npt

from pygeoinf.linear_forms import LinearForm

from pygeoinf3d.geometries.box.domain import BoxDomain
from pygeoinf3d.providers.tensor_product import TensorProductProvider
from pygeoinf3d.spaces.sobolev import Sobolev3D, SpectralSobolev3D
from pygeoinf3d.spaces.forms import SobolevKernel3D
from pygeoinf3d.core.functions import Function3D


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def unit_box():
    return BoxDomain(np.zeros(3), np.ones(3))


@pytest.fixture
def dir_provider(unit_box):
    return TensorProductProvider(
        unit_box, truncation=(3, 3, 3), bc_type='dirichlet'
    )


@pytest.fixture
def neu_provider(unit_box):
    return TensorProductProvider(
        unit_box, truncation=(3, 3, 3), bc_type='neumann'
    )


@pytest.fixture
def dir_space(dir_provider):
    return SpectralSobolev3D(dir_provider, s=1.0, kappa=1.0)


@pytest.fixture
def neu_space(neu_provider):
    return SpectralSobolev3D(neu_provider, s=1.5, kappa=0.5)


@pytest.fixture
def s0_space(dir_provider):
    """s=0 degenerates to an unweighted (scaled) L² space."""
    return SpectralSobolev3D(dir_provider, s=0.0, kappa=1.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit_vector(n: int, k: int) -> np.ndarray:
    """Unit vector e_k in R^n."""
    e = np.zeros(n)
    e[k] = 1.0
    return e


# ---------------------------------------------------------------------------
# Construction and properties
# ---------------------------------------------------------------------------

class TestSpectralSobolev3DConstruction:

    def test_is_hilbert_space(self, dir_space):
        from pygeoinf import HilbertSpace
        assert isinstance(dir_space, HilbertSpace)

    def test_dim_equals_provider_dim(self, dir_space, dir_provider):
        assert dir_space.dim == dir_provider.dim

    def test_s_property(self, dir_space):
        assert dir_space.s == pytest.approx(1.0)

    def test_kappa_property(self, dir_space):
        assert dir_space.kappa == pytest.approx(1.0)

    def test_function_domain_is_box(self, dir_space, unit_box):
        assert dir_space.function_domain is unit_box

    def test_provider_property(self, dir_space, dir_provider):
        assert dir_space.provider is dir_provider

    def test_eigenvalues_shape(self, dir_space):
        lams = dir_space.eigenvalues
        assert lams.shape == (dir_space.dim,)

    def test_eigenvalues_positive(self, dir_space):
        lams = dir_space.eigenvalues
        assert np.all(lams > 0.0)

    def test_eigenvalues_sorted(self, dir_space):
        lams = dir_space.eigenvalues
        assert np.all(lams[1:] >= lams[:-1])

    def test_weights_shape(self, dir_space):
        w = dir_space.weights
        assert w.shape == (dir_space.dim,)

    def test_weights_formula(self, dir_space):
        """w_k = (kappa^2 + lambda_k)^s."""
        lams = dir_space.eigenvalues
        expected = (dir_space.kappa**2 + lams) ** dir_space.s
        npt.assert_allclose(dir_space.weights, expected, rtol=1e-12)

    def test_weights_all_positive(self, dir_space):
        assert np.all(dir_space.weights > 0.0)

    def test_neumann_space_dim(self, neu_space, neu_provider):
        assert neu_space.dim == neu_provider.dim

    def test_negative_s_raises(self, dir_provider):
        with pytest.raises(ValueError, match="s must be >= 0"):
            SpectralSobolev3D(dir_provider, s=-0.1)

    def test_negative_kappa_raises(self, dir_provider):
        with pytest.raises(ValueError, match="kappa must be >= 0"):
            SpectralSobolev3D(dir_provider, s=1.0, kappa=-0.5)

    def test_equality_same_instance(self, dir_space):
        assert dir_space == dir_space

    def test_equality_different_s(self, dir_provider):
        s1 = SpectralSobolev3D(dir_provider, s=1.0, kappa=1.0)
        s2 = SpectralSobolev3D(dir_provider, s=2.0, kappa=1.0)
        assert s1 != s2

    def test_equality_different_kappa(self, dir_provider):
        s1 = SpectralSobolev3D(dir_provider, s=1.0, kappa=1.0)
        s2 = SpectralSobolev3D(dir_provider, s=1.0, kappa=2.0)
        assert s1 != s2


# ---------------------------------------------------------------------------
# Component roundtrip
# ---------------------------------------------------------------------------

class TestComponentRoundtrip:

    def test_to_components_identity(self, dir_space):
        rng = np.random.default_rng(7)
        c = rng.standard_normal(dir_space.dim)
        npt.assert_array_equal(dir_space.to_components(c), c)

    def test_from_components_identity(self, dir_space):
        rng = np.random.default_rng(8)
        c = rng.standard_normal(dir_space.dim)
        npt.assert_array_equal(dir_space.from_components(c), c)

    def test_roundtrip_preserves_values(self, dir_space):
        rng = np.random.default_rng(9)
        c = rng.standard_normal(dir_space.dim)
        npt.assert_array_equal(
            dir_space.from_components(dir_space.to_components(c)), c
        )

    def test_zero_element_shape(self, dir_space):
        z = dir_space.zero
        assert isinstance(z, np.ndarray)
        assert z.shape == (dir_space.dim,)

    def test_zero_element_is_zero(self, dir_space):
        npt.assert_array_equal(dir_space.zero, 0.0)


# ---------------------------------------------------------------------------
# Weighted inner product
# ---------------------------------------------------------------------------

class TestWeightedInnerProduct:

    def test_orthonormality_diagonal(self, dir_space):
        """<e_k, e_k>_Hs = weights[k]."""
        N = dir_space.dim
        w = dir_space.weights
        for k in range(N):
            ek = _unit_vector(N, k)
            ip_kk = dir_space.inner_product(ek, ek)
            npt.assert_allclose(ip_kk, w[k], rtol=1e-12)

    def test_cross_terms_zero(self, dir_space):
        """<e_k, e_j>_Hs = 0 for k != j."""
        N = dir_space.dim
        for k in range(min(N, 3)):
            for j in range(min(N, 3)):
                if k == j:
                    continue
                ek = _unit_vector(N, k)
                ej = _unit_vector(N, j)
                npt.assert_allclose(dir_space.inner_product(ek, ej), 0.0,
                                    atol=1e-15)

    def test_self_adjoint(self, dir_space):
        """<c1, c2> = <c2, c1>."""
        rng = np.random.default_rng(42)
        c1 = rng.standard_normal(dir_space.dim)
        c2 = rng.standard_normal(dir_space.dim)
        npt.assert_allclose(
            dir_space.inner_product(c1, c2),
            dir_space.inner_product(c2, c1),
            rtol=1e-12,
        )

    def test_positive_definite(self, dir_space):
        """<c, c> > 0 for c != 0."""
        rng = np.random.default_rng(43)
        c = rng.standard_normal(dir_space.dim)
        assert dir_space.inner_product(c, c) > 0.0

    def test_positivity_zero(self, dir_space):
        assert dir_space.inner_product(dir_space.zero, dir_space.zero) == 0.0

    def test_s0_equals_l2_scaled(self, dir_provider):
        """s=0 with kappa=1 gives all weights=1;
        inner product equals dot product."""
        # With s=0, weights are (kappa^2 + lam)^0 = 1 for all modes.
        space = SpectralSobolev3D(dir_provider, s=0.0, kappa=1.0)
        rng = np.random.default_rng(44)
        c1 = rng.standard_normal(space.dim)
        c2 = rng.standard_normal(space.dim)
        npt.assert_allclose(
            space.inner_product(c1, c2),
            np.dot(c1, c2),
            rtol=1e-12,
        )


# ---------------------------------------------------------------------------
# Dual mapping
# ---------------------------------------------------------------------------

class TestDualMapping:

    def test_to_dual_returns_linear_form(self, dir_space):
        c = _unit_vector(dir_space.dim, 0)
        fp = dir_space.to_dual(c)
        assert isinstance(fp, LinearForm)

    def test_to_dual_components(self, dir_space):
        """to_dual(e_k).components[k] = weight[k], rest zero."""
        N = dir_space.dim
        w = dir_space.weights
        for k in range(min(N, 4)):
            ek = _unit_vector(N, k)
            fp = dir_space.to_dual(ek)
            expected = w[k] * ek
            npt.assert_allclose(fp.components, expected, rtol=1e-12)

    def test_to_dual_evaluation_matches_inner_product(self, dir_space):
        """to_dual(c1)(c2) == inner_product(c1, c2)."""
        rng = np.random.default_rng(55)
        c1 = rng.standard_normal(dir_space.dim)
        c2 = rng.standard_normal(dir_space.dim)
        fp = dir_space.to_dual(c1)
        npt.assert_allclose(
            fp(c2), dir_space.inner_product(c1, c2), rtol=1e-12
        )

    def test_from_dual_roundtrip(self, dir_space):
        """from_dual(to_dual(c)) == c."""
        rng = np.random.default_rng(66)
        c = rng.standard_normal(dir_space.dim)
        fp = dir_space.to_dual(c)
        c_back = dir_space.from_dual(fp)
        npt.assert_allclose(c_back, c, rtol=1e-12)

    def test_from_dual_wrong_type_raises(self, dir_space):
        with pytest.raises(TypeError):
            dir_space.from_dual("not a linear form")

    def test_from_dual_foreign_space_raises(self, dir_provider):
        left = SpectralSobolev3D(dir_provider, s=1.0, kappa=1.0)
        right = SpectralSobolev3D(dir_provider, s=1.0, kappa=1.0)
        dual = left.to_dual(_unit_vector(left.dim, 0))
        with pytest.raises(TypeError, match="exact SpectralSobolev3D"):
            right.from_dual(dual)

    def test_duality_product_matches_inner_product(self, dir_space):
        """duality_product(to_dual(c1), c2) == inner_product(c1, c2)."""
        rng = np.random.default_rng(77)
        c1 = rng.standard_normal(dir_space.dim)
        c2 = rng.standard_normal(dir_space.dim)
        ip = dir_space.inner_product(c1, c2)
        dp = dir_space.duality_product(dir_space.to_dual(c1), c2)
        npt.assert_allclose(ip, dp, rtol=1e-12)


# ---------------------------------------------------------------------------
# Spectral conversion
# ---------------------------------------------------------------------------

class TestSpectralConversion:

    def test_project_eigenfunction_unit_vector(
        self, dir_space, dir_provider, unit_box
    ):
        """Projecting phi_k onto basis should recover e_k
        (up to quadrature error)."""
        for k in range(min(dir_space.dim, 3)):
            phi_k = dir_provider.get_eigenfunction(k)
            c = dir_space.project(phi_k, quad_resolution=25)
            # Should be ~1 at index k and ~0 elsewhere
            msg = f"phi_{k} projection failed at index {k}"
            npt.assert_allclose(c[k], 1.0, atol=1e-3, err_msg=msg)
            others = np.delete(c, k)
            npt.assert_allclose(others, 0.0, atol=1e-3,
                                err_msg=f"phi_{k} projection has cross-terms")

    def test_to_function_first_eigenfunction(
        self, dir_space, dir_provider, unit_box
    ):
        """to_function(e_0)(x) should match phi_0(x) pointwise."""
        N = dir_space.dim
        e0 = _unit_vector(N, 0)
        f_reconstructed = dir_space.to_function(e0)
        phi0 = dir_provider.get_eigenfunction(0)

        pts = unit_box.structured_mesh(15)
        vals_rec = np.asarray(f_reconstructed(pts), dtype=float)
        vals_ref = np.asarray(phi0(pts), dtype=float)
        npt.assert_allclose(vals_rec, vals_ref, atol=1e-12)

    def test_to_function_zero_gives_zero(self, dir_space, unit_box):
        z = dir_space.zero
        f = dir_space.to_function(z)
        pts = unit_box.structured_mesh(10)
        npt.assert_allclose(f(pts), 0.0, atol=1e-15)

    def test_project_recover_roundtrip(
        self, dir_space, dir_provider, unit_box
    ):
        """project(to_function(c)) ≈ c for spectral modes within truncation."""
        rng = np.random.default_rng(88)
        c = rng.standard_normal(dir_space.dim)
        f = dir_space.to_function(c)
        c_back = dir_space.project(f, quad_resolution=25)
        npt.assert_allclose(c_back, c, atol=1e-2)


# ---------------------------------------------------------------------------
# Trace readiness
# ---------------------------------------------------------------------------

class TestTraceReadiness:

    def test_supports_trace_high_s(self, dir_space):
        """s=1.0 > 0.5 should support trace."""
        assert dir_space.supports_trace is True

    def test_supports_trace_s_above_threshold(self, dir_provider):
        space = SpectralSobolev3D(dir_provider, s=0.6)
        assert space.supports_trace is True

    def test_supports_trace_at_boundary(self, dir_provider):
        """s=0.5 is exactly at the threshold → should NOT support trace."""
        space = SpectralSobolev3D(dir_provider, s=0.5)
        assert space.supports_trace is False

    def test_supports_trace_low_s(self, dir_provider):
        space = SpectralSobolev3D(dir_provider, s=0.3)
        assert space.supports_trace is False

    def test_supports_trace_s0(self, s0_space):
        assert s0_space.supports_trace is False

    def test_assert_trace_allowed_passes(self, dir_space):
        dir_space.assert_trace_allowed()  # should not raise

    def test_assert_trace_allowed_raises_for_low_s(self, dir_provider):
        space = SpectralSobolev3D(dir_provider, s=0.4)
        with pytest.raises(RuntimeError, match="Trace is not defined"):
            space.assert_trace_allowed()


# ---------------------------------------------------------------------------
# BoxDomain sample_boundary and outward_normal (trace hooks)
# ---------------------------------------------------------------------------

class TestBoxBoundaryHooks:

    def test_sample_boundary_shape(self, unit_box):
        pts = unit_box.sample_boundary(100, rng=np.random.default_rng(1))
        assert pts.shape == (100, 3)

    def test_sample_boundary_all_on_boundary(self, unit_box):
        pts = unit_box.sample_boundary(200, rng=np.random.default_rng(2))
        for p in pts:
            assert unit_box.boundary.contains(p, rtol=1e-6), \
                f"Point {p} not on boundary"

    def test_sample_boundary_reproducible(self, unit_box):
        pts1 = unit_box.sample_boundary(50, rng=np.random.default_rng(42))
        pts2 = unit_box.sample_boundary(50, rng=np.random.default_rng(42))
        npt.assert_array_equal(pts1, pts2)

    def test_sample_boundary_shifted_box(self):
        box = BoxDomain(np.array([1.0, 2.0, -1.0]), np.array([2.0, 4.0, 1.0]))
        pts = box.sample_boundary(200, rng=np.random.default_rng(3))
        assert pts.shape == (200, 3)
        for p in pts:
            assert box.boundary.contains(p, rtol=1e-6)

    def test_outward_normal_shape(self, unit_box):
        pts = unit_box.sample_boundary(50, rng=np.random.default_rng(4))
        norms = unit_box.outward_normal(pts)
        assert norms.shape == (50, 3)

    def test_outward_normal_unit_vectors(self, unit_box):
        pts = unit_box.sample_boundary(100, rng=np.random.default_rng(5))
        norms = unit_box.outward_normal(pts)
        magnitudes = np.linalg.norm(norms, axis=1)
        npt.assert_allclose(magnitudes, 1.0, atol=1e-12)

    def test_outward_normal_face_correct_x_min(self, unit_box):
        """Points on x=0 face should have normal [-1, 0, 0]."""
        rng = np.random.default_rng(6)
        ys = rng.uniform(0.0, 1.0, 20)
        zs = rng.uniform(0.0, 1.0, 20)
        pts = np.column_stack([np.zeros(20), ys, zs])
        norms = unit_box.outward_normal(pts, component='x_min')
        expected = np.tile([-1.0, 0.0, 0.0], (20, 1))
        npt.assert_allclose(norms, expected, atol=1e-12)

    def test_outward_normal_face_correct_x_max(self, unit_box):
        rng = np.random.default_rng(7)
        ys = rng.uniform(0.0, 1.0, 20)
        zs = rng.uniform(0.0, 1.0, 20)
        pts = np.column_stack([np.ones(20), ys, zs])
        norms = unit_box.outward_normal(pts, component='x_max')
        expected = np.tile([+1.0, 0.0, 0.0], (20, 1))
        npt.assert_allclose(norms, expected, atol=1e-12)

    def test_outward_normal_invalid_component_raises(self, unit_box):
        pts = np.array([[0.5, 0.5, 0.5]])
        with pytest.raises(ValueError, match="Unknown boundary component"):
            unit_box.outward_normal(pts, component='bad_face')


# ---------------------------------------------------------------------------
# pygeoinf compatibility smoke tests
# ---------------------------------------------------------------------------

class TestPygeoinf3DCompatibility:

    def test_zero_is_ndarray(self, dir_space):
        z = dir_space.zero
        assert isinstance(z, np.ndarray)
        assert z.shape == (dir_space.dim,)

    def test_random_is_ndarray(self, dir_space):
        np.random.seed(0)
        c = dir_space.random()
        assert isinstance(c, np.ndarray)
        assert c.shape == (dir_space.dim,)

    def test_add_subtract(self, dir_space):
        rng = np.random.default_rng(10)
        c1 = rng.standard_normal(dir_space.dim)
        c2 = rng.standard_normal(dir_space.dim)
        res = dir_space.add(c1, c2)
        npt.assert_array_equal(res, c1 + c2)

    def test_multiply(self, dir_space):
        rng = np.random.default_rng(11)
        c = rng.standard_normal(dir_space.dim)
        npt.assert_array_equal(dir_space.multiply(3.0, c), 3.0 * c)

    def test_copy_independent(self, dir_space):
        rng = np.random.default_rng(12)
        c = rng.standard_normal(dir_space.dim)
        c2 = dir_space.copy(c)
        c2[0] += 99.0
        assert c[0] != c2[0]

    def test_linear_operator_on_sobolev(self, dir_space):
        """A LinearOperator scaling by 2 works on Sobolev3D elements."""
        from pygeoinf.linear_operators import LinearOperator
        op = LinearOperator(dir_space, dir_space, lambda c: 2.0 * c,
                            adjoint_mapping=lambda c: 2.0 * c)
        rng = np.random.default_rng(13)
        c = rng.standard_normal(dir_space.dim)
        result = op(c)
        npt.assert_allclose(result, 2.0 * c, rtol=1e-12)


# ---------------------------------------------------------------------------
# Degeneracy validation (review fix #2)
# ---------------------------------------------------------------------------

class TestSpectralSobolev3DDegeneracyValidation:

    def test_neumann_kappa0_s_positive_raises(self, neu_provider):
        """Neumann + kappa=0 + s>0 is degenerate and must raise ValueError."""
        with pytest.raises(ValueError, match="degenerate"):
            SpectralSobolev3D(neu_provider, s=1.0, kappa=0.0)

    def test_neumann_kappa0_s_small_positive_raises(self, neu_provider):
        """Any positive s triggers the guard, not just large values."""
        with pytest.raises(ValueError, match="degenerate"):
            SpectralSobolev3D(neu_provider, s=0.1, kappa=0.0)

    def test_neumann_kappa0_s0_ok(self, neu_provider):
        """s=0 gives all weights=1 regardless of kappa; should succeed."""
        space = SpectralSobolev3D(neu_provider, s=0.0, kappa=0.0)
        npt.assert_allclose(space.weights, 1.0)

    def test_dirichlet_kappa0_positive_s_ok(self, dir_provider):
        """Dirichlet has no zero eigenvalue so kappa=0 + s>0 is valid."""
        space = SpectralSobolev3D(dir_provider, s=1.0, kappa=0.0)
        assert np.all(space.weights > 0.0)

    def test_neumann_kappa_positive_ok(self, neu_provider):
        """Neumann + kappa>0 + any s should succeed."""
        space = SpectralSobolev3D(neu_provider, s=2.0, kappa=0.01)
        assert np.all(space.weights > 0.0)


# ---------------------------------------------------------------------------
# is_element override (review fix #4)
# ---------------------------------------------------------------------------

class TestIsElement:

    def test_valid_array_is_element(self, dir_space):
        c = np.zeros(dir_space.dim)
        assert dir_space.is_element(c) is True

    def test_random_array_is_element(self, dir_space):
        rng = np.random.default_rng(20)
        c = rng.standard_normal(dir_space.dim)
        assert dir_space.is_element(c) is True

    def test_too_long_not_element(self, dir_space):
        c = np.zeros(dir_space.dim + 1)
        assert dir_space.is_element(c) is False

    def test_too_short_not_element(self, dir_space):
        c = np.zeros(max(dir_space.dim - 1, 0))
        assert dir_space.is_element(c) is False

    def test_2d_array_not_element(self, dir_space):
        c = np.zeros((dir_space.dim, 1))
        assert dir_space.is_element(c) is False

    def test_string_not_element(self, dir_space):
        assert dir_space.is_element("not an array") is False


# ---------------------------------------------------------------------------
# TensorProductProvider.box public property (review fix #6)
# ---------------------------------------------------------------------------

class TestProviderBoxProperty:

    def test_box_property_returns_domain(self, dir_provider, unit_box):
        assert dir_provider.box is unit_box

    def test_function_domain_uses_box_property(
        self, dir_space, dir_provider, unit_box
    ):
        """function_domain must use provider.box, not provider._box."""
        assert dir_space.function_domain is dir_provider.box
        assert dir_space.function_domain is unit_box


# ===========================================================================
# New continuous-first Sobolev3D tests
# ===========================================================================


@pytest.fixture
def sobolev(dir_provider):
    """Continuous-first Sobolev3D (s=1.0, kappa=1.0)."""
    return Sobolev3D(dir_provider, s=1.0, kappa=1.0)


class TestSobolev3DConstruction:
    """Construction and basic properties of continuous-first Sobolev3D."""

    def test_dim_matches_provider_dim(self, sobolev, dir_provider):
        assert sobolev.dim == dir_provider.dim

    def test_s_property(self, sobolev):
        assert sobolev.s == 1.0

    def test_kappa_property(self, sobolev):
        assert sobolev.kappa == 1.0

    def test_quad_resolution_default(self, dir_provider):
        space = Sobolev3D(dir_provider, s=1.0)
        assert space.quad_resolution == 20

    def test_quad_resolution_custom(self, dir_provider):
        space = Sobolev3D(dir_provider, s=1.0, quad_resolution=30)
        assert space.quad_resolution == 30

    def test_function_domain_is_box(self, sobolev, unit_box):
        assert sobolev.function_domain is unit_box

    def test_eigenvalues_shape(self, sobolev, dir_provider):
        assert sobolev.eigenvalues.shape == (dir_provider.dim,)

    def test_weights_shape(self, sobolev, dir_provider):
        assert sobolev.weights.shape == (dir_provider.dim,)

    def test_negative_s_raises(self, dir_provider):
        with pytest.raises(ValueError, match="s must be"):
            Sobolev3D(dir_provider, s=-1.0)

    def test_negative_kappa_raises(self, dir_provider):
        with pytest.raises(ValueError, match="kappa"):
            Sobolev3D(dir_provider, s=1.0, kappa=-0.5)

    def test_equality(self, sobolev, dir_provider):
        other = Sobolev3D(dir_provider, s=1.0, kappa=1.0)
        assert sobolev == other

    def test_inequality_different_s(self, sobolev, dir_provider):
        other = Sobolev3D(dir_provider, s=2.0, kappa=1.0)
        assert sobolev != other


class TestSobolev3DIsElement:
    """is_element logic for continuous-first Sobolev3D."""

    def test_function3d_on_same_domain_is_element(self, sobolev, unit_box):
        f = Function3D(unit_box, evaluate_callable=lambda x: np.ones(len(x)))
        assert sobolev.is_element(f)

    def test_eigenfunction_is_element(self, sobolev, dir_provider):
        phi0 = dir_provider.get_eigenfunction(0)
        assert sobolev.is_element(phi0)

    def test_function3d_wrong_domain_not_element(self, sobolev):
        other_box = BoxDomain(np.zeros(3), 2 * np.ones(3))
        f = Function3D(other_box, evaluate_callable=lambda x: np.ones(len(x)))
        assert not sobolev.is_element(f)

    def test_ndarray_not_element(self, sobolev, dir_provider):
        c = np.zeros(dir_provider.dim)
        assert not sobolev.is_element(c)

    def test_string_not_element(self, sobolev):
        assert not sobolev.is_element("not a function")


class TestSobolev3DDualMapping:
    """Riesz map and its inverse for continuous-first Sobolev3D."""

    def test_to_dual_returns_sobolev_kernel(self, sobolev, dir_provider):
        phi0 = dir_provider.get_eigenfunction(0)
        dual = sobolev.to_dual(phi0)
        assert isinstance(dual, SobolevKernel3D)

    def test_from_dual_recovers_kernel_pointwise(
        self, sobolev, dir_provider, unit_box
    ):
        phi0 = dir_provider.get_eigenfunction(0)
        dual = sobolev.to_dual(phi0)
        f_back = sobolev.from_dual(dual)
        pts = unit_box.structured_mesh(10)
        npt.assert_allclose(f_back(pts), phi0(pts), atol=1e-12)

    def test_from_dual_wrong_type_raises(self, sobolev):
        with pytest.raises(TypeError):
            sobolev.from_dual("not a SobolevKernel3D")

    def test_to_dual_wrong_type_raises(self, sobolev):
        with pytest.raises(TypeError):
            sobolev.to_dual("not a Function3D")

    def test_to_dual_wrong_domain_raises(self, sobolev):
        other_box = BoxDomain(np.zeros(3), 2 * np.ones(3))
        f = Function3D(other_box, evaluate_callable=lambda x: np.ones(len(x)))
        with pytest.raises(ValueError):
            sobolev.to_dual(f)

    def test_from_dual_foreign_space_raises(self, dir_provider):
        left = Sobolev3D(dir_provider, s=1.0, kappa=1.0)
        right = Sobolev3D(dir_provider, s=1.0, kappa=1.0)
        phi0 = dir_provider.get_eigenfunction(0)
        dual = left.to_dual(phi0)
        with pytest.raises(ValueError, match="different Sobolev3D"):
            right.from_dual(dual)

    def test_to_dual_evaluates_inner_product(
        self, sobolev, dir_provider, unit_box
    ):
        """to_dual(phi_i)(phi_j) == inner_product(phi_i, phi_j)."""
        phi0 = dir_provider.get_eigenfunction(0)
        phi1 = dir_provider.get_eigenfunction(1)
        ip = sobolev.inner_product(phi0, phi1)
        dp = sobolev.to_dual(phi0)(phi1)
        npt.assert_allclose(dp, ip, rtol=1e-10)


class TestSobolev3DInnerProduct:
    """Weighted H^s inner product for continuous-first Sobolev3D."""

    def test_eigenfunction_self_ip(self, sobolev, dir_provider):
        """<phi_0, phi_0>_Hs = (kappa^2 + lambda_0)^s to 0.5% tolerance."""
        phi0 = dir_provider.get_eigenfunction(0)
        lam0 = dir_provider.get_eigenvalue(0)
        expected = (sobolev.kappa ** 2 + lam0) ** sobolev.s
        ip = sobolev.inner_product(phi0, phi0)
        npt.assert_allclose(ip, expected, rtol=5e-3)

    def test_orthogonality(self, sobolev, dir_provider):
        """<phi_0, phi_1>_Hs approx 0."""
        phi0 = dir_provider.get_eigenfunction(0)
        phi1 = dir_provider.get_eigenfunction(1)
        ip = sobolev.inner_product(phi0, phi1)
        npt.assert_allclose(ip, 0.0, atol=1e-3)

    def test_symmetry(self, sobolev, dir_provider):
        phi0 = dir_provider.get_eigenfunction(0)
        phi1 = dir_provider.get_eigenfunction(1)
        npt.assert_allclose(
            sobolev.inner_product(phi0, phi1),
            sobolev.inner_product(phi1, phi0),
            rtol=1e-10,
        )

    def test_positive_definite(self, sobolev, dir_provider):
        phi0 = dir_provider.get_eigenfunction(0)
        assert sobolev.inner_product(phi0, phi0) > 0.0

    def test_norm_nonnegative(self, sobolev, dir_provider):
        phi0 = dir_provider.get_eigenfunction(0)
        assert sobolev.norm(phi0) >= 0.0


class TestSobolev3DComponents:
    """to_components / from_components for continuous-first Sobolev3D."""

    def test_to_components_returns_ndarray(self, sobolev, dir_provider):
        phi0 = dir_provider.get_eigenfunction(0)
        c = sobolev.to_components(phi0)
        assert isinstance(c, np.ndarray)
        assert c.shape == (dir_provider.dim,)

    def test_to_components_eigenfunction_dominant(
        self, sobolev, dir_provider
    ):
        """to_components(phi_k)[k] ~= 1, others ~= 0."""
        for k in range(min(3, dir_provider.dim)):
            phi_k = dir_provider.get_eigenfunction(k)
            c = sobolev.to_components(phi_k)
            npt.assert_allclose(c[k], 1.0, atol=1e-3)
            mask = np.ones(dir_provider.dim, dtype=bool)
            mask[k] = False
            npt.assert_allclose(c[mask], 0.0, atol=1e-3)

    def test_from_components_roundtrip(
        self, sobolev, dir_provider, unit_box
    ):
        """from_components(to_components(phi0)) ~= phi0 pointwise."""
        phi0 = dir_provider.get_eigenfunction(0)
        c = sobolev.to_components(phi0)
        f = sobolev.from_components(c)
        pts = unit_box.structured_mesh(10)
        npt.assert_allclose(f(pts), phi0(pts), atol=5e-3)


class TestSobolev3DVectorOps:
    """Vector space operations for continuous-first Sobolev3D."""

    def test_zero_is_function3d(self, sobolev):
        assert isinstance(sobolev.zero, Function3D)

    def test_zero_evaluates_to_zero(self, sobolev, unit_box):
        pts = unit_box.structured_mesh(5)
        npt.assert_allclose(sobolev.zero(pts), 0.0, atol=1e-15)

    def test_random_returns_function3d(self, sobolev):
        np.random.seed(0)
        assert isinstance(sobolev.random(), Function3D)

    def test_add_function3d(self, sobolev, dir_provider, unit_box):
        phi0 = dir_provider.get_eigenfunction(0)
        phi1 = dir_provider.get_eigenfunction(1)
        f_sum = sobolev.add(phi0, phi1)
        pts = unit_box.structured_mesh(8)
        npt.assert_allclose(f_sum(pts), phi0(pts) + phi1(pts), atol=1e-12)

    def test_multiply_function3d(self, sobolev, dir_provider, unit_box):
        phi0 = dir_provider.get_eigenfunction(0)
        f_scaled = sobolev.multiply(3.0, phi0)
        pts = unit_box.structured_mesh(8)
        npt.assert_allclose(f_scaled(pts), 3.0 * phi0(pts), atol=1e-12)

    def test_negative_function3d(self, sobolev, dir_provider, unit_box):
        phi0 = dir_provider.get_eigenfunction(0)
        f_neg = sobolev.negative(phi0)
        pts = unit_box.structured_mesh(8)
        npt.assert_allclose(f_neg(pts), -phi0(pts), atol=1e-12)


class TestSobolev3DTraceReadiness:
    """Trace guard for continuous-first Sobolev3D."""

    def test_supports_trace_s1(self, sobolev):
        assert sobolev.supports_trace is True

    def test_supports_trace_low_s(self, dir_provider):
        space = Sobolev3D(dir_provider, s=0.3)
        assert space.supports_trace is False

    def test_assert_trace_allowed_passes(self, sobolev):
        sobolev.assert_trace_allowed()

    def test_assert_trace_allowed_raises(self, dir_provider):
        space = Sobolev3D(dir_provider, s=0.4)
        with pytest.raises(RuntimeError, match="Trace is not defined"):
            space.assert_trace_allowed()


class TestSobolev3DDegeneracyValidation:
    """Degeneracy guard for continuous-first Sobolev3D."""

    def test_neumann_kappa0_s_positive_raises(self, neu_provider):
        with pytest.raises(ValueError, match="degenerate"):
            Sobolev3D(neu_provider, s=1.0, kappa=0.0)

    def test_dirichlet_kappa0_positive_s_ok(self, dir_provider):
        space = Sobolev3D(dir_provider, s=1.0, kappa=0.0)
        assert np.all(space.weights > 0.0)

    def test_neumann_kappa_positive_ok(self, neu_provider):
        space = Sobolev3D(neu_provider, s=2.0, kappa=0.01)
        assert np.all(space.weights > 0.0)
