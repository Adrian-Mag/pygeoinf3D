"""Tests for pygeoinf3d.sampling.kl_sampler.KLSampler3D.

Covers:
- Construction validation (type, n_modes bounds)
- sample() returns Function3D with correct shape
- Samples are distinct (different random draws)
- Mean function contribution (statistical, fixed seed)
- covariance_factor: type, domain/codomain, eigenfunction reconstruction
- Zero-mode Neumann suppression in covariance_factor
"""

from __future__ import annotations

import pytest
import numpy as np
import numpy.testing as npt

from pygeoinf import EuclideanSpace
from pygeoinf.linear_operators import LinearOperator

from pygeoinf3d.geometries.box.domain import BoxDomain
from pygeoinf3d.providers.tensor_product import TensorProductProvider
from pygeoinf3d.spaces.lebesgue import Lebesgue3D
from pygeoinf3d.operators.inverse_laplacian_box import InverseBoxLaplacian
from pygeoinf3d.operators.laplacian_box import BoxLaplacian
from pygeoinf3d.operators.base import SpectralOperator3D
from pygeoinf3d.core.functions import Function3D
from pygeoinf3d.sampling import KLSampler3D


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def unit_box():
    return BoxDomain(np.zeros(3), np.ones(3))


@pytest.fixture
def space(unit_box):
    return Lebesgue3D(unit_box)


@pytest.fixture
def dirichlet_provider(unit_box):
    return TensorProductProvider(unit_box, truncation=(3, 3, 3), bc_type='dirichlet')


@pytest.fixture
def neumann_provider(unit_box):
    return TensorProductProvider(unit_box, truncation=(3, 3, 3), bc_type='neumann')


@pytest.fixture
def inv_lap_dirichlet(space, dirichlet_provider):
    return InverseBoxLaplacian(space, dirichlet_provider, quad_resolution=20)


@pytest.fixture
def inv_lap_neumann_pseudo(space, neumann_provider):
    return InverseBoxLaplacian(
        space, neumann_provider,
        null_space_handling='pseudoinverse',
        quad_resolution=20,
    )


@pytest.fixture
def sampler_dirichlet(inv_lap_dirichlet):
    return KLSampler3D(inv_lap_dirichlet, n_modes=5, rng=np.random.default_rng(0))


@pytest.fixture
def sampler_neumann(inv_lap_neumann_pseudo):
    return KLSampler3D(inv_lap_neumann_pseudo, n_modes=5, rng=np.random.default_rng(1))


def _eval_pts(box, n=100):
    rng = np.random.default_rng(999)
    lo = box.bounding_box().low
    hi = box.bounding_box().high
    return rng.uniform(lo, hi, size=(n, 3))


# ---------------------------------------------------------------------------
# TestKLSamplerConstruction
# ---------------------------------------------------------------------------

class TestKLSamplerConstruction:

    def test_construct_dirichlet(self, sampler_dirichlet):
        assert sampler_dirichlet is not None

    def test_construct_neumann(self, sampler_neumann):
        assert sampler_neumann is not None

    def test_n_modes_property(self, sampler_dirichlet):
        assert sampler_dirichlet.n_modes == 5

    def test_space_property(self, sampler_dirichlet, space):
        assert sampler_dirichlet.space == space

    def test_eigenvalues_shape(self, sampler_dirichlet):
        assert sampler_dirichlet.eigenvalues.shape == (5,)

    def test_bad_n_modes_zero_raises(self, inv_lap_dirichlet):
        with pytest.raises(ValueError, match="n_modes"):
            KLSampler3D(inv_lap_dirichlet, n_modes=0)

    def test_bad_n_modes_negative_raises(self, inv_lap_dirichlet):
        with pytest.raises(ValueError, match="n_modes"):
            KLSampler3D(inv_lap_dirichlet, n_modes=-1)

    def test_n_modes_exceeds_operator_raises(self, inv_lap_dirichlet):
        too_many = inv_lap_dirichlet.n_modes + 1
        with pytest.raises(ValueError, match="n_modes"):
            KLSampler3D(inv_lap_dirichlet, n_modes=too_many)

    def test_wrong_operator_type_raises(self, space, dirichlet_provider):
        # BoxLaplacian is a SpectralOperator3D, so use a plain LinearOperator
        from pygeoinf.linear_operators import LinearOperator
        from pygeoinf import EuclideanSpace
        dummy = LinearOperator(EuclideanSpace(3), EuclideanSpace(3), lambda x: x)
        with pytest.raises(TypeError, match="SpectralOperator3D"):
            KLSampler3D(dummy, n_modes=1)

    def test_can_construct_from_box_laplacian(self, space, dirichlet_provider):
        """BoxLaplacian is also a SpectralOperator3D and can be wrapped."""
        lap = BoxLaplacian(space, dirichlet_provider)
        sampler = KLSampler3D(lap, n_modes=4)
        assert sampler is not None


# ---------------------------------------------------------------------------
# TestKLSamplerSample
# ---------------------------------------------------------------------------

class TestKLSamplerSample:

    def test_sample_returns_function3d(self, sampler_dirichlet):
        s = sampler_dirichlet.sample()
        assert isinstance(s, Function3D)

    def test_sample_batch_shape(self, unit_box, sampler_dirichlet):
        s = sampler_dirichlet.sample()
        pts = _eval_pts(unit_box, n=80)
        vals = s(pts)
        assert vals.shape == (80,)

    def test_sample_single_point(self, sampler_dirichlet):
        s = sampler_dirichlet.sample()
        val = s(np.array([0.5, 0.5, 0.5]))
        assert np.isscalar(val) or val.shape == ()

    def test_samples_are_distinct(self, unit_box, inv_lap_dirichlet):
        # Use fresh rng so two consecutive samples are different
        sampler = KLSampler3D(inv_lap_dirichlet, n_modes=5, rng=np.random.default_rng(42))
        pts = _eval_pts(unit_box, n=20)
        s1 = sampler.sample()(pts)
        s2 = sampler.sample()(pts)
        assert not np.allclose(s1, s2)

    def test_samples_list_length(self, sampler_dirichlet):
        slist = sampler_dirichlet.samples(7)
        assert len(slist) == 7
        assert all(isinstance(s, Function3D) for s in slist)

    def test_mean_is_added_statistically(self, unit_box, inv_lap_dirichlet):
        """E[sample(x_0)] ≈ mean(x_0) — statistical test with fixed seed."""
        mean_val = 3.14
        mean_fn = Function3D(
            unit_box,
            evaluate_callable=lambda x: (
                np.full(len(x), mean_val) if np.asarray(x).ndim == 2 else float(mean_val)
            ),
        )
        sampler = KLSampler3D(
            inv_lap_dirichlet, n_modes=4, mean=mean_fn, rng=np.random.default_rng(77)
        )
        center = np.array([[0.5, 0.5, 0.5]])
        avg = np.mean([sampler.sample()(center)[0] for _ in range(300)])
        # Generous tolerance: std-error ≈ std / sqrt(300).  Std of each mode
        # at the center is O(sqrt(lam_k)*phi_k(center)) which is small.
        npt.assert_allclose(avg, mean_val, atol=0.5)


# ---------------------------------------------------------------------------
# TestKLSamplerCovarianceFactor
# ---------------------------------------------------------------------------

class TestKLSamplerCovarianceFactor:

    def test_is_linear_operator(self, sampler_dirichlet):
        cf = sampler_dirichlet.covariance_factor
        assert isinstance(cf, LinearOperator)

    def test_domain_is_euclidean(self, sampler_dirichlet):
        cf = sampler_dirichlet.covariance_factor
        assert isinstance(cf.domain, EuclideanSpace)
        assert cf.domain.dim == sampler_dirichlet.n_modes

    def test_codomain_is_space(self, sampler_dirichlet, space):
        cf = sampler_dirichlet.covariance_factor
        assert cf.codomain == space

    def test_eigenfunction_reconstruction_mode0(self, unit_box, inv_lap_dirichlet):
        """L(e_0) = sqrt(lam_0) * phi_0."""
        sampler = KLSampler3D(inv_lap_dirichlet, n_modes=5)
        cf = sampler.covariance_factor
        K = sampler.n_modes
        e0 = np.zeros(K); e0[0] = 1.0
        result = cf(e0)
        pts = _eval_pts(unit_box, n=50)
        phi0 = inv_lap_dirichlet.get_eigenfunction(0)
        lam0 = inv_lap_dirichlet.get_eigenvalue(0)
        npt.assert_allclose(
            result(pts), np.sqrt(lam0) * phi0(pts), rtol=1e-6, atol=1e-10
        )

    def test_eigenfunction_reconstruction_mode1(self, unit_box, inv_lap_dirichlet):
        """L(e_1) = sqrt(lam_1) * phi_1."""
        sampler = KLSampler3D(inv_lap_dirichlet, n_modes=5)
        cf = sampler.covariance_factor
        K = sampler.n_modes
        e1 = np.zeros(K); e1[1] = 1.0
        result = cf(e1)
        pts = _eval_pts(unit_box, n=50)
        phi1 = inv_lap_dirichlet.get_eigenfunction(1)
        lam1 = inv_lap_dirichlet.get_eigenvalue(1)
        npt.assert_allclose(
            result(pts), np.sqrt(lam1) * phi1(pts), rtol=1e-6, atol=1e-10
        )

    def test_zero_mode_neumann_vanishes_in_cf(self, unit_box, inv_lap_neumann_pseudo):
        """With Neumann pseudoinverse, mode 0 has lam_0=0 → L(e_0) = 0."""
        sampler = KLSampler3D(inv_lap_neumann_pseudo, n_modes=5)
        cf = sampler.covariance_factor
        K = sampler.n_modes
        e0 = np.zeros(K); e0[0] = 1.0
        result = cf(e0)
        pts = _eval_pts(unit_box, n=50)
        npt.assert_allclose(result(pts), 0.0, atol=1e-10)

    def test_property_returns_new_instance_each_call(self, sampler_dirichlet):
        """covariance_factor is a property that returns fresh LinearOperator."""
        cf1 = sampler_dirichlet.covariance_factor
        cf2 = sampler_dirichlet.covariance_factor
        # Different objects, but same mathematical behaviour
        assert cf1 is not cf2
        K = sampler_dirichlet.n_modes
        w = np.ones(K)
        # Applying both to same input should give identical results
        pts = np.array([[0.5, 0.5, 0.5]])
        npt.assert_allclose(cf1(w)(pts), cf2(w)(pts), rtol=1e-12)
