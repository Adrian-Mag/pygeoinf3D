"""Integration tests: KLSampler3D with GeneralizedBoxLaplacian.

Verifies that GeneralizedBoxLaplacian works correctly with KLSampler3D,
that the tensor-product fast path is preserved for all Laplacian wrapper
types (BoxLaplacian, InverseBoxLaplacian), and that negative-power operators
behave correctly under sampling.

Phase 4 of pygeoinf3d-generalized-box-laplacian-plan.md — TEST-ONLY.
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from pygeoinf3d.geometries.box.domain import BoxDomain
from pygeoinf3d.providers.tensor_product import TensorProductProvider
from pygeoinf3d.spaces.lebesgue import Lebesgue3D
from pygeoinf3d.operators.generalized_box_laplacian import GeneralizedBoxLaplacian
from pygeoinf3d.operators.laplacian_box import BoxLaplacian
from pygeoinf3d.operators.inverse_laplacian_box import InverseBoxLaplacian
from pygeoinf3d.core.functions import Function3D
from pygeoinf3d.sampling import KLSampler3D


# ---------------------------------------------------------------------------
# Module-level fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def unit_box():
    return BoxDomain(np.zeros(3), np.ones(3))


@pytest.fixture(scope="module")
def space(unit_box):
    return Lebesgue3D(unit_box)


@pytest.fixture(scope="module")
def dirichlet_provider(unit_box):
    return TensorProductProvider(unit_box, truncation=(3, 3, 3), bc_type='dirichlet')


@pytest.fixture(scope="module")
def neumann_provider(unit_box):
    return TensorProductProvider(unit_box, truncation=(3, 3, 3), bc_type='neumann')


# ---------------------------------------------------------------------------
# TestKLWithGeneralizedBoxLaplacian
# ---------------------------------------------------------------------------

class TestKLWithGeneralizedBoxLaplacian:

    def test_kl_accepts_generalized_negative_power(self, space, dirichlet_provider):
        """KLSampler3D should accept GeneralizedBoxLaplacian(power=-1) without error."""
        gen_op = GeneralizedBoxLaplacian(space, dirichlet_provider, power=-1)
        sampler = KLSampler3D(gen_op, n_modes=5, rng=np.random.default_rng(0))
        assert sampler.n_modes == 5

    def test_kl_negative_power_has_tensor_product_fast_path(
        self, space, dirichlet_provider
    ):
        """GeneralizedBoxLaplacian(power=-1) should expose _provider so the fast path
        is active."""
        gen_op = GeneralizedBoxLaplacian(space, dirichlet_provider, power=-1)
        sampler = KLSampler3D(gen_op, n_modes=5, rng=np.random.default_rng(1))
        assert sampler.has_tensor_product_structure is True

    def test_kl_sample_returns_function3d(self, space, dirichlet_provider):
        """sample() from a GeneralizedBoxLaplacian sampler returns a Function3D."""
        gen_op = GeneralizedBoxLaplacian(space, dirichlet_provider, power=-1)
        sampler = KLSampler3D(gen_op, n_modes=5, rng=np.random.default_rng(2))
        fn = sampler.sample()
        assert isinstance(fn, Function3D)

    def test_kl_zero_eigenvalue_mode_contributes_zero(self, space, neumann_provider):
        """Neumann zero mode is suppressed: eigenvalue[0] == 0 and sqrt-eigenvalue[0] == 0."""
        gen_op = GeneralizedBoxLaplacian(space, neumann_provider, power=-1)
        npt.assert_allclose(gen_op.get_eigenvalue(0), 0.0, atol=0.0)
        sampler = KLSampler3D(gen_op, n_modes=5, rng=np.random.default_rng(3))
        npt.assert_allclose(sampler._sqrt_eigenvalues[0], 0.0, atol=0.0)

    def test_kl_positive_power_nonnegative_eigenvalues(self, space, dirichlet_provider):
        """GeneralizedBoxLaplacian(power=2) eigenvalues are all >= 0."""
        gen_op = GeneralizedBoxLaplacian(space, dirichlet_provider, power=2)
        sampler = KLSampler3D(gen_op, n_modes=5, rng=np.random.default_rng(4))
        assert np.all(sampler.eigenvalues >= 0)

    def test_kl_negative_power_sample_values_finite(self, space, dirichlet_provider):
        """Samples from GeneralizedBoxLaplacian(power=-1) are finite everywhere."""
        gen_op = GeneralizedBoxLaplacian(space, dirichlet_provider, power=-1)
        sampler = KLSampler3D(gen_op, n_modes=5, rng=np.random.default_rng(42))
        fn = sampler.sample()
        rng = np.random.default_rng(999)
        pts = rng.uniform(0.0, 1.0, size=(20, 3))
        vals = fn(pts)
        assert np.all(np.isfinite(vals))

    def test_kl_power_neg1_matches_inverse_boxlaplacian_sampling(
        self, space, dirichlet_provider
    ):
        """GeneralizedBoxLaplacian(power=-1) and InverseBoxLaplacian(pseudoinverse)
        are equivalent operators: same eigenvalues, same eigenfunctions, same samples."""
        gen_op = GeneralizedBoxLaplacian(
            space, dirichlet_provider, power=-1, alpha=1.0
        )
        inv_lap = InverseBoxLaplacian(
            space, dirichlet_provider, null_space_handling='pseudoinverse'
        )

        # Verify eigenvalues match for first 5 modes
        for i in range(5):
            npt.assert_allclose(
                gen_op.get_eigenvalue(i),
                inv_lap.get_eigenvalue(i),
                rtol=1e-12,
                err_msg=f"eigenvalue mismatch at mode {i}",
            )

        # Same seed -> same xi values -> same sample
        seed = 77
        sampler_gen = KLSampler3D(gen_op, n_modes=5, rng=np.random.default_rng(seed))
        sampler_inv = KLSampler3D(inv_lap, n_modes=5, rng=np.random.default_rng(seed))

        fn_gen = sampler_gen.sample()
        fn_inv = sampler_inv.sample()

        rng = np.random.default_rng(1234)
        pts = rng.uniform(0.0, 1.0, size=(20, 3))
        npt.assert_allclose(fn_gen(pts), fn_inv(pts), rtol=1e-12)

    def test_kl_inverse_boxlaplacian_still_has_fast_path(
        self, space, dirichlet_provider
    ):
        """After Phase 3 refactor, InverseBoxLaplacian still exposes _provider so
        KLSampler3D detects tensor-product structure."""
        inv_lap = InverseBoxLaplacian(
            space, dirichlet_provider, null_space_handling='pseudoinverse'
        )
        sampler = KLSampler3D(inv_lap, n_modes=5, rng=np.random.default_rng(5))
        assert sampler.has_tensor_product_structure is True

    def test_kl_box_laplacian_still_has_fast_path(self, space, dirichlet_provider):
        """After Phase 3, BoxLaplacian inherits _provider from GeneralizedBoxLaplacian
        and is visible to KLSampler3D."""
        box_lap = BoxLaplacian(space, dirichlet_provider)
        sampler = KLSampler3D(box_lap, n_modes=5, rng=np.random.default_rng(6))
        assert sampler.has_tensor_product_structure is True

    def test_kl_power_0_identity_sample_finite(self, space, dirichlet_provider):
        """GeneralizedBoxLaplacian(power=0) is the identity: all eigenvalues == 1,
        samples are finite."""
        gen_op = GeneralizedBoxLaplacian(space, dirichlet_provider, power=0)
        # All eigenvalues should be 1.0
        for i in range(5):
            npt.assert_allclose(gen_op.get_eigenvalue(i), 1.0, rtol=1e-15)
        sampler = KLSampler3D(gen_op, n_modes=5, rng=np.random.default_rng(7))
        fn = sampler.sample()
        rng = np.random.default_rng(8888)
        pts = rng.uniform(0.0, 1.0, size=(20, 3))
        vals = fn(pts)
        assert np.all(np.isfinite(vals))
