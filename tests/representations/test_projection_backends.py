"""Tests for Phase 2: Gram-Aware Backend Infrastructure.

Covers:
- OrthonormalProjectionBackend: moments == coefficients, synthesis round-trip,
  gram_matrix is identity, coefficient_metric is standard Euclidean.
- GramProjectionBackend: gram matrix shape/symmetry/positive-definiteness,
  moment assembly, coefficient solve consistency, synthesis round-trip,
  coefficient_metric is non-Euclidean for non-orthogonal bases.
- Consistency: G @ to_components(f) ≈ moments(f) for both backends.
- LebesgueRepresentation3D exposes coefficient_metric(), gram_matrix(),
  and moments().

All tests use small mock providers with simple polynomial functions on [0,1]³.
The mock domain integrates using a dense deterministic midpoint grid (G=40³
points) to achieve rtol ≈ 1e-3 with analytic tolerances for
degree-≤2 polynomials.
"""

from __future__ import annotations

import numpy as np
import pytest

from pygeoinf import EuclideanSpace, Ball, MassWeightedHilbertSpace

from pygeoinf3d.core.config import AxisAlignedBoundingBox
from pygeoinf3d.core.region import Region3D
from pygeoinf3d.core.domain_base import Domain3D
from pygeoinf3d.core.boundary import Boundary3D
from pygeoinf3d.core.functions import Function3D


# ---------------------------------------------------------------------------
# Mock domain: deterministic midpoint rule on [0,1]^3
# ---------------------------------------------------------------------------

class _MockBoundary(Boundary3D):
    def contains(self, point: np.ndarray) -> bool:
        return False


class _GridDomain(Domain3D):
    """Unit cube [0,1]^3 with a deterministic midpoint-rule quadrature.

    Uses a 30³ = 27,000-point regular grid.  Resulting integration errors
    for polynomials of total degree ≤ 2 are below 1e-4.
    """

    _G = 40  # grid resolution per axis; degree-2 error is about 3e-4

    def __init__(self, region: Region3D) -> None:
        super().__init__(region)
        g = self._G
        lin = np.linspace(0.5 / g, 1.0 - 0.5 / g, g)
        xx, yy, zz = np.meshgrid(lin, lin, lin, indexing="ij")
        self._pts: np.ndarray = np.stack(
            [xx.ravel(), yy.ravel(), zz.ravel()], axis=-1
        )
        self._w: np.ndarray = np.full(g ** 3, 1.0 / g ** 3)

    def sample_interior(self, n: int, *, rng=None) -> np.ndarray:
        rng = np.random.default_rng(rng)
        return rng.uniform(0.0, 1.0, size=(n, 3))

    def integrate_volume(self, f, *, method=None, **kwargs) -> float:
        return float(np.dot(f(self._pts), self._w))

    @property
    def boundary(self) -> Boundary3D:
        return _MockBoundary(self)


def _make_grid_domain() -> _GridDomain:
    space_3d = EuclideanSpace(3)
    ball = Ball(space_3d, np.array([0.5, 0.5, 0.5]), 0.5)
    bbox = AxisAlignedBoundingBox(low=np.zeros(3), high=np.ones(3))
    return _GridDomain(Region3D(ball, bounding_box=bbox))


# ---------------------------------------------------------------------------
# Orthonormal mock provider for [0,1]^3 with uniform measure
#
# Basis functions:
#   phi_0(x) = 1                            (constant)
#   phi_1(x) = sqrt(3) * (2 x[0] - 1)      (Legendre-like on x[0])
#   phi_2(x) = sqrt(3) * (2 x[1] - 1)      (Legendre-like on x[1])
#
# Orthonormality under uniform measure on [0,1]^3:
#   ∫ phi_i^2 = 1,   ∫ phi_i phi_j = 0 for i≠j.
# ---------------------------------------------------------------------------

class _OrthonormalProvider:
    """Mock provider exposing 3 orthonormal polynomial basis functions."""

    _DIM = 3

    def __init__(self, domain: Domain3D) -> None:
        self.domain = domain
        self._basis = [
            Function3D(
                domain,
                evaluate_callable=lambda x: (
                    np.ones(len(x)) if np.ndim(x) == 2 else 1.0
                ),
            ),
            Function3D(
                domain,
                evaluate_callable=lambda x: (
                    np.sqrt(3.0) * (2.0 * x[..., 0] - 1.0)
                ),
            ),
            Function3D(
                domain,
                evaluate_callable=lambda x: (
                    np.sqrt(3.0) * (2.0 * x[..., 1] - 1.0)
                ),
            ),
        ]

    @property
    def dim(self) -> int:
        return self._DIM

    def get_basis_function(self, i: int) -> Function3D:
        return self._basis[i]


# ---------------------------------------------------------------------------
# Non-orthogonal mock provider for [0,1]^3 with uniform measure
#
# Basis functions:
#   psi_0(x) = 1           psi_1(x) = x[0]
#
# Gram matrix (analytic):
#   G = [[1,    1/2 ],
#        [1/2,  1/3 ]]
#
# For f(x) = x[0]:
#   b = [1/2, 1/3]
#   c = inv(G) @ b = [[4, -6], [-6, 12]] @ [1/2, 1/3] = [0, 1]
#   synthesis → c[0]*psi_0 + c[1]*psi_1 = x[0]  (exact)
# ---------------------------------------------------------------------------

class _NonOrthogonalProvider:
    """Mock provider exposing 2 non-orthogonal polynomial basis functions."""

    _DIM = 2

    def __init__(self, domain: Domain3D) -> None:
        self.domain = domain
        self._basis = [
            Function3D(
                domain,
                evaluate_callable=lambda x: (
                    np.ones(len(x)) if np.ndim(x) == 2 else 1.0
                ),
            ),
            Function3D(domain, evaluate_callable=lambda x: x[..., 0]),
        ]

    @property
    def dim(self) -> int:
        return self._DIM

    def get_basis_function(self, i: int) -> Function3D:
        return self._basis[i]


# Analytic Gram for non-orthogonal case
_GRAM_ANALYTIC = np.array([[1.0, 0.5], [0.5, 1.0 / 3.0]])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def grid_domain() -> _GridDomain:
    return _make_grid_domain()


@pytest.fixture(scope="module")
def lebesgue_space(grid_domain):
    from pygeoinf3d.spaces.lebesgue import Lebesgue3D
    return Lebesgue3D(grid_domain)


@pytest.fixture(scope="module")
def orthonormal_provider(grid_domain) -> _OrthonormalProvider:
    return _OrthonormalProvider(grid_domain)


@pytest.fixture(scope="module")
def non_orthogonal_provider(grid_domain) -> _NonOrthogonalProvider:
    return _NonOrthogonalProvider(grid_domain)


@pytest.fixture(scope="module")
def orthonormal_backend():
    from pygeoinf3d.representations.backends import (
        OrthonormalProjectionBackend,
    )
    return OrthonormalProjectionBackend()


@pytest.fixture(scope="module")
def gram_backend():
    from pygeoinf3d.representations.backends import GramProjectionBackend
    return GramProjectionBackend()


# ---------------------------------------------------------------------------
# Helper: f(x) = x[0]
# ---------------------------------------------------------------------------

def _f_x0(domain: Domain3D) -> Function3D:
    return Function3D(domain, evaluate_callable=lambda x: x[..., 0])


def _f_const(domain: Domain3D, v: float = 1.0) -> Function3D:
    return Function3D(
        domain,
        evaluate_callable=lambda x: (
            np.full(len(x), v) if np.ndim(x) == 2 else v
        ),
    )


# ===========================================================================
# Tests: gram.py utilities
# ===========================================================================

class TestAssembleGram:
    def test_orthonormal_gram_approx_identity(
        self, lebesgue_space, orthonormal_provider
    ):
        from pygeoinf3d.representations.gram import assemble_gram
        G = assemble_gram(lebesgue_space, orthonormal_provider)
        assert G.shape == (3, 3)
        np.testing.assert_allclose(G, np.eye(3), atol=1e-3,
                                   err_msg="Orthonormal Gram should be ≈ I")

    def test_non_orthogonal_gram_values(
        self, lebesgue_space, non_orthogonal_provider
    ):
        from pygeoinf3d.representations.gram import assemble_gram
        G = assemble_gram(lebesgue_space, non_orthogonal_provider)
        assert G.shape == (2, 2)
        np.testing.assert_allclose(G, _GRAM_ANALYTIC, atol=1e-3,
                                   err_msg="Non-orthogonal Gram mismatch")

    def test_gram_is_symmetric(self, lebesgue_space, non_orthogonal_provider):
        from pygeoinf3d.representations.gram import assemble_gram
        G = assemble_gram(lebesgue_space, non_orthogonal_provider)
        np.testing.assert_allclose(G, G.T, atol=1e-12,
                                   err_msg="Gram should be symmetric")

    def test_gram_is_positive_definite(
        self, lebesgue_space, non_orthogonal_provider
    ):
        from pygeoinf3d.representations.gram import assemble_gram
        G = assemble_gram(lebesgue_space, non_orthogonal_provider)
        eigvals = np.linalg.eigvalsh(G)
        assert np.all(eigvals > 0), f"Non-PD Gram: eigenvalues = {eigvals}"


class TestGramOperators:
    def test_returns_two_operators(self):
        from pygeoinf3d.representations.gram import gram_operators
        G = _GRAM_ANALYTIC
        M, M_inv = gram_operators(G)
        assert M is not None
        assert M_inv is not None

    def test_mass_operator_applies_gram(self):
        from pygeoinf3d.representations.gram import gram_operators
        G = _GRAM_ANALYTIC
        M, _ = gram_operators(G)
        c = np.array([1.0, 0.0])
        result = M(c)
        np.testing.assert_allclose(result, G @ c, atol=1e-12)

    def test_inverse_mass_operator_inverts(self):
        from pygeoinf3d.representations.gram import gram_operators
        G = _GRAM_ANALYTIC
        _, M_inv = gram_operators(G)
        b = np.array([0.5, 0.25])
        c = M_inv(b)
        np.testing.assert_allclose(G @ c, b, atol=1e-10,
                                   err_msg="M_inv should solve G c = b")

    def test_round_trip_mass_inverse(self):
        from pygeoinf3d.representations.gram import gram_operators
        G = _GRAM_ANALYTIC
        M, M_inv = gram_operators(G)
        c = np.array([2.0, -1.0])
        np.testing.assert_allclose(M_inv(M(c)), c, atol=1e-10)


# ===========================================================================
# Tests: OrthonormalProjectionBackend
# ===========================================================================

class TestOrthonormalProjectionBackend:
    # ------------------------------------------------------------------
    # moments
    # ------------------------------------------------------------------
    def test_moments_constant_function(
        self, lebesgue_space, orthonormal_provider, orthonormal_backend
    ):
        """∫ phi_0 * 1 = 1,  ∫ phi_k * 1 = 0 for k>0."""
        f = _f_const(lebesgue_space.function_domain, 1.0)
        b = orthonormal_backend.moments(
            lebesgue_space, orthonormal_provider, f
        )
        assert b.shape == (3,)
        np.testing.assert_allclose(b[0], 1.0, atol=1e-3)  # phi_0 = 1
        np.testing.assert_allclose(b[1], 0.0, atol=1e-3)
        np.testing.assert_allclose(b[2], 0.0, atol=1e-3)

    def test_moments_equals_to_components_orthonormal(
        self, lebesgue_space, orthonormal_provider, orthonormal_backend
    ):
        """For orthonormal basis, moments and to_components must agree."""
        f = _f_x0(lebesgue_space.function_domain)
        b = orthonormal_backend.moments(
            lebesgue_space, orthonormal_provider, f
        )
        c = orthonormal_backend.to_components(
            lebesgue_space, orthonormal_provider, f
        )
        np.testing.assert_allclose(
            c,
            b,
            atol=1e-12,
            err_msg="Orthonormal: moments == to_components",
        )

    # ------------------------------------------------------------------
    # gram_matrix
    # ------------------------------------------------------------------
    def test_gram_matrix_is_identity(
        self, lebesgue_space, orthonormal_provider, orthonormal_backend
    ):
        G = orthonormal_backend.gram_matrix(
            lebesgue_space, orthonormal_provider
        )
        n = orthonormal_provider.dim
        np.testing.assert_allclose(
            G,
            np.eye(n),
            atol=1e-12,
            err_msg="Orthonormal backend gram_matrix must be I_n",
        )

    # ------------------------------------------------------------------
    # solve_coefficients
    # ------------------------------------------------------------------
    def test_solve_coefficients_is_identity_transform(
        self, lebesgue_space, orthonormal_provider, orthonormal_backend
    ):
        b = np.array([0.5, 0.1, -0.3])
        G = orthonormal_backend.gram_matrix(
            lebesgue_space, orthonormal_provider
        )
        c = orthonormal_backend.solve_coefficients(G, b)
        np.testing.assert_allclose(c, b, atol=1e-12)

    # ------------------------------------------------------------------
    # coefficient_metric
    # ------------------------------------------------------------------
    def test_coefficient_metric_is_mass_weighted_hilbert_space(
        self, lebesgue_space, orthonormal_provider, orthonormal_backend
    ):
        metric = orthonormal_backend.coefficient_metric(
            lebesgue_space, orthonormal_provider
        )
        assert isinstance(metric, MassWeightedHilbertSpace)

    def test_coefficient_metric_euclidean_inner_product(
        self, lebesgue_space, orthonormal_provider, orthonormal_backend
    ):
        """Orthonormal metric inner product equals the Euclidean one."""
        metric = orthonormal_backend.coefficient_metric(
            lebesgue_space, orthonormal_provider
        )
        c1 = np.array([1.0, 0.0, 0.0])
        c2 = np.array([0.0, 1.0, 0.0])
        # Must be 0 (orthogonal standard basis vectors)
        np.testing.assert_allclose(
            metric.inner_product(c1, c2), 0.0, atol=1e-12
        )
        np.testing.assert_allclose(
            metric.inner_product(c1, c1), 1.0, atol=1e-12
        )

    # ------------------------------------------------------------------
    # from_components / synthesis round-trip
    # ------------------------------------------------------------------
    def test_synthesis_constant_coefficients(
        self, lebesgue_space, orthonormal_provider, orthonormal_backend
    ):
        """c = [2, 0, 0] → f_syn(x) ≈ 2 everywhere."""
        c = np.array([2.0, 0.0, 0.0])
        f_syn = orthonormal_backend.from_components(
            lebesgue_space, orthonormal_provider, c
        )
        test_pts = np.array(
            [[0.1, 0.2, 0.3], [0.5, 0.5, 0.5], [0.9, 0.8, 0.7]]
        )
        vals = f_syn(test_pts)
        np.testing.assert_allclose(vals, 2.0, atol=1e-12)

    def test_round_trip_to_components_from_components(
        self, lebesgue_space, orthonormal_provider, orthonormal_backend
    ):
        """Synthesis then analysis should recover orthonormal coefficients."""
        c_in = np.array([0.7, -0.3, 0.1])
        f_syn = orthonormal_backend.from_components(
            lebesgue_space, orthonormal_provider, c_in
        )
        c_out = orthonormal_backend.to_components(
            lebesgue_space, orthonormal_provider, f_syn
        )
        np.testing.assert_allclose(
            c_out,
            c_in,
            atol=1e-3,
            err_msg="Orthonormal round-trip",
        )


# ===========================================================================
# Tests: GramProjectionBackend
# ===========================================================================

class TestGramProjectionBackend:
    # ------------------------------------------------------------------
    # gram_matrix
    # ------------------------------------------------------------------
    def test_gram_shape(
        self, lebesgue_space, non_orthogonal_provider, gram_backend
    ):
        G = gram_backend.gram_matrix(lebesgue_space, non_orthogonal_provider)
        assert G.shape == (2, 2)

    def test_gram_symmetry(
        self, lebesgue_space, non_orthogonal_provider, gram_backend
    ):
        G = gram_backend.gram_matrix(lebesgue_space, non_orthogonal_provider)
        np.testing.assert_allclose(G, G.T, atol=1e-12)

    def test_gram_positive_definite(
        self, lebesgue_space, non_orthogonal_provider, gram_backend
    ):
        G = gram_backend.gram_matrix(lebesgue_space, non_orthogonal_provider)
        assert np.all(np.linalg.eigvalsh(G) > 0)

    def test_gram_matches_analytic(
        self, lebesgue_space, non_orthogonal_provider, gram_backend
    ):
        G = gram_backend.gram_matrix(lebesgue_space, non_orthogonal_provider)
        np.testing.assert_allclose(G, _GRAM_ANALYTIC, atol=1e-3)

    # ------------------------------------------------------------------
    # moments
    # ------------------------------------------------------------------
    def test_moments_shape(
        self, lebesgue_space, non_orthogonal_provider, gram_backend
    ):
        f = _f_x0(lebesgue_space.function_domain)
        b = gram_backend.moments(lebesgue_space, non_orthogonal_provider, f)
        assert b.shape == (2,)

    def test_moments_f_x0_analytic(
        self, lebesgue_space, non_orthogonal_provider, gram_backend
    ):
        """For f=x0 and psi_0=1, psi_1=x0:  b_0=1/2, b_1=1/3."""
        f = _f_x0(lebesgue_space.function_domain)
        b = gram_backend.moments(lebesgue_space, non_orthogonal_provider, f)
        np.testing.assert_allclose(b[0], 0.5, atol=1e-3,
                                   err_msg="b_0 = ∫ 1·x₀ dV = 1/2")
        np.testing.assert_allclose(b[1], 1.0 / 3.0, atol=1e-3,
                                   err_msg="b_1 = ∫ x₀·x₀ dV = 1/3")

    # ------------------------------------------------------------------
    # solve_coefficients
    # ------------------------------------------------------------------
    def test_solve_coefficients_exact(self, gram_backend):
        """Analytic solve: G [[1,1/2],[1/2,1/3]] c = [1/2,1/3] → c = [0,1]."""
        G = _GRAM_ANALYTIC
        b = np.array([0.5, 1.0 / 3.0])
        c = gram_backend.solve_coefficients(G, b)
        np.testing.assert_allclose(c, np.array([0.0, 1.0]), atol=1e-10)

    # ------------------------------------------------------------------
    # to_components
    # ------------------------------------------------------------------
    def test_to_components_f_x0(
        self, lebesgue_space, non_orthogonal_provider, gram_backend
    ):
        """f = x₀ is exactly in span{psi_0, psi_1}: c = [0, 1]."""
        f = _f_x0(lebesgue_space.function_domain)
        c = gram_backend.to_components(
            lebesgue_space, non_orthogonal_provider, f
        )
        np.testing.assert_allclose(
            c,
            np.array([0.0, 1.0]),
            atol=1e-3,
            err_msg="f=x₀ should have coefficients [0,1]",
        )

    # ------------------------------------------------------------------
    # from_components / synthesis
    # ------------------------------------------------------------------
    def test_from_components_synthesises_correctly(
        self, lebesgue_space, non_orthogonal_provider, gram_backend
    ):
        """c = [0, 1] → synthesis = x₀ everywhere."""
        c = np.array([0.0, 1.0])
        f_syn = gram_backend.from_components(
            lebesgue_space, non_orthogonal_provider, c
        )
        pts = np.array([[0.0, 0.5, 0.5], [0.3, 0.1, 0.9], [1.0, 0.0, 0.5]])
        expected = pts[:, 0]
        np.testing.assert_allclose(f_syn(pts), expected, atol=1e-12)

    # ------------------------------------------------------------------
    # coefficient_metric
    # ------------------------------------------------------------------
    def test_coefficient_metric_is_mass_weighted_hilbert_space(
        self, lebesgue_space, non_orthogonal_provider, gram_backend
    ):
        metric = gram_backend.coefficient_metric(
            lebesgue_space, non_orthogonal_provider
        )
        assert isinstance(metric, MassWeightedHilbertSpace)

    def test_coefficient_metric_inner_product_uses_gram(
        self, lebesgue_space, non_orthogonal_provider, gram_backend
    ):
        """<c1, c2>_metric = c1 @ G @ c2 (Gram-weighted inner product)."""
        metric = gram_backend.coefficient_metric(
            lebesgue_space, non_orthogonal_provider
        )
        G = _GRAM_ANALYTIC
        c1 = np.array([1.0, 0.0])
        c2 = np.array([0.0, 1.0])
        expected = float(c1 @ G @ c2)   # = G[0,1] = 0.5
        np.testing.assert_allclose(
            metric.inner_product(c1, c2), expected, atol=1e-10
        )

    def test_coefficient_metric_differs_from_euclidean(
        self, lebesgue_space, non_orthogonal_provider, gram_backend
    ):
        """The Gram metric inner product must differ from the dot product."""
        metric = gram_backend.coefficient_metric(
            lebesgue_space, non_orthogonal_provider
        )
        c1 = np.array([1.0, 0.0])
        c2 = np.array([0.0, 1.0])
        gram_ip = metric.inner_product(c1, c2)
        euclidean_ip = float(np.dot(c1, c2))   # = 0
        # Gram inner product is G[0,1] = 0.5, which differs from Euclidean (0)
        assert not np.isclose(gram_ip, euclidean_ip, atol=1e-8), (
            "Non-orthogonal coefficient metric should not equal "
            "Euclidean metric"
        )

    def test_coefficient_metric_dimension(
        self, lebesgue_space, non_orthogonal_provider, gram_backend
    ):
        metric = gram_backend.coefficient_metric(
            lebesgue_space, non_orthogonal_provider
        )
        assert metric.dim == non_orthogonal_provider.dim


# ===========================================================================
# Tests: Gram-moment consistency
# ===========================================================================

class TestMomentConsistency:
    def test_gram_times_components_equals_moments_orthonormal(
        self, lebesgue_space, orthonormal_provider, orthonormal_backend
    ):
        """For all backends: G @ to_components(f) ≈ moments(f)."""
        f = _f_x0(lebesgue_space.function_domain)
        b = orthonormal_backend.moments(
            lebesgue_space, orthonormal_provider, f
        )
        c = orthonormal_backend.to_components(
            lebesgue_space, orthonormal_provider, f
        )
        G = orthonormal_backend.gram_matrix(
            lebesgue_space, orthonormal_provider
        )
        np.testing.assert_allclose(
            G @ c,
            b,
            atol=1e-12,
            err_msg="G @ c must equal b (orthonormal case, G=I)",
        )

    def test_gram_times_components_equals_moments_non_orthogonal(
        self, lebesgue_space, non_orthogonal_provider, gram_backend
    ):
        """G @ to_components(f) ≈ moments(f) for GramProjectionBackend."""
        f = _f_x0(lebesgue_space.function_domain)
        b = gram_backend.moments(lebesgue_space, non_orthogonal_provider, f)
        c = gram_backend.to_components(
            lebesgue_space, non_orthogonal_provider, f
        )
        G = gram_backend.gram_matrix(lebesgue_space, non_orthogonal_provider)
        np.testing.assert_allclose(
            G @ c,
            b,
            atol=1e-3,
            err_msg="G @ c must equal b (non-orthogonal)",
        )

    def test_orthonormal_backend_gram_caching(
        self, lebesgue_space, orthonormal_provider, orthonormal_backend
    ):
        """Repeated identity Gram calls should have no side effects."""
        G1 = orthonormal_backend.gram_matrix(
            lebesgue_space, orthonormal_provider
        )
        G2 = orthonormal_backend.gram_matrix(
            lebesgue_space, orthonormal_provider
        )
        np.testing.assert_array_equal(G1, G2)

    def test_gram_backend_gram_caching(
        self, lebesgue_space, non_orthogonal_provider, gram_backend
    ):
        """GramProjectionBackend should return the same matrix on repeats."""
        G1 = gram_backend.gram_matrix(lebesgue_space, non_orthogonal_provider)
        G2 = gram_backend.gram_matrix(lebesgue_space, non_orthogonal_provider)
        np.testing.assert_array_equal(G1, G2)


# ===========================================================================
# Tests: LebesgueRepresentation3D Phase 2 façade
# ===========================================================================

class TestLebesgueRepresentationPhase2:
    """Verify LebesgueRepresentation3D delegates Phase-2 methods to backend."""

    @pytest.fixture
    def non_orthogonal_rep(self, lebesgue_space, non_orthogonal_provider):
        from pygeoinf3d.representations.lebesgue import (
            LebesgueRepresentation3D,
        )
        from pygeoinf3d.representations.backends import GramProjectionBackend
        backend = GramProjectionBackend()
        return LebesgueRepresentation3D(
            lebesgue_space, non_orthogonal_provider, backend
        )

    @pytest.fixture
    def orthonormal_rep(self, lebesgue_space, orthonormal_provider):
        from pygeoinf3d.representations.lebesgue import (
            LebesgueRepresentation3D,
        )
        from pygeoinf3d.representations.backends import (
            OrthonormalProjectionBackend,
        )
        backend = OrthonormalProjectionBackend()
        return LebesgueRepresentation3D(
            lebesgue_space, orthonormal_provider, backend
        )

    def test_representation_exposes_coefficient_metric(
        self, non_orthogonal_rep
    ):
        metric = non_orthogonal_rep.coefficient_metric()
        assert isinstance(metric, MassWeightedHilbertSpace)

    def test_representation_exposes_gram_matrix(self, non_orthogonal_rep):
        G = non_orthogonal_rep.gram_matrix()
        np.testing.assert_allclose(G, _GRAM_ANALYTIC, atol=1e-3)

    def test_representation_exposes_moments(
        self, lebesgue_space, non_orthogonal_rep
    ):
        f = _f_x0(lebesgue_space.function_domain)
        b = non_orthogonal_rep.moments(f)
        np.testing.assert_allclose(b[0], 0.5, atol=1e-3)
        np.testing.assert_allclose(b[1], 1.0 / 3.0, atol=1e-3)

    def test_representation_coefficient_metric_dimension(
        self, non_orthogonal_rep
    ):
        metric = non_orthogonal_rep.coefficient_metric()
        assert metric.dim == non_orthogonal_rep.dim

    def test_base_class_coefficient_metric_raises_without_backend_support(
        self, lebesgue_space
    ):
        """A minimal LebesgueRepresentation3D with a stub backend should raise.
        """
        from pygeoinf3d.representations.lebesgue import (
            LebesgueRepresentation3D,
        )

        class _StubBackend:
            def to_components(self, space, provider, f):
                return np.zeros(provider.dim)

            def from_components(self, space, provider, c):
                return Function3D(
                    space.function_domain,
                    evaluate_callable=lambda x: (
                        np.zeros(len(x)) if np.ndim(x) == 2 else 0.0
                    ),
                )

        class _StubProvider:
            dim = 2

            def get_basis_function(self, i):
                return Function3D(
                    lebesgue_space.function_domain,
                    evaluate_callable=lambda x: (
                        np.zeros(len(x)) if np.ndim(x) == 2 else 0.0
                    ),
                )

        rep = LebesgueRepresentation3D(
            lebesgue_space, _StubProvider(), _StubBackend()
        )
        with pytest.raises(NotImplementedError):
            rep.coefficient_metric()

    def test_orthonormal_rep_coefficient_metric(self, orthonormal_rep):
        metric = orthonormal_rep.coefficient_metric()
        assert isinstance(metric, MassWeightedHilbertSpace)
        # For orthonormal basis, metric is standard Euclidean
        c = np.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(metric.inner_product(c, c), 1.0, atol=1e-12)

    def test_to_components_and_from_components_via_rep(
        self, lebesgue_space, non_orthogonal_rep
    ):
        """Round-trip through representation analysis and synthesis."""
        f = _f_x0(lebesgue_space.function_domain)
        c = non_orthogonal_rep.to_components(f)
        np.testing.assert_allclose(c, np.array([0.0, 1.0]), atol=1e-3)
        f_syn = non_orthogonal_rep.from_components(c)
        pts = np.array([[0.2, 0.4, 0.6], [0.8, 0.1, 0.3]])
        np.testing.assert_allclose(f_syn(pts), pts[:, 0], atol=1e-12)
