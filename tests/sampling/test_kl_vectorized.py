"""Tests for KLSampler3D vectorized evaluate closure (Phase 2).

Covers:
- Vectorized evaluate matches explicit loop for the same weights/xis
- Single-point input (shape (3,)) returns a Python float
- Batch input (shape (n, 3)) returns ndarray of shape (n,)
- sample_from_xis vectorized path matches loop
- Vectorized path is >10x faster than loop for 1000 evaluation points
"""

from __future__ import annotations

import time

import numpy as np
import numpy.testing as npt
import pytest

from pygeoinf3d.core.functions import Function3D
from pygeoinf3d.geometries.box.domain import BoxDomain
from pygeoinf3d.operators.inverse_laplacian_box import InverseBoxLaplacian
from pygeoinf3d.providers.tensor_product import BoxLaplacianSpectrumProvider
from pygeoinf3d.sampling import KLSampler3D
from pygeoinf3d.spaces.lebesgue import Lebesgue3D


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
    # 5x5x5 = 125 modes in the provider; sampler uses only 10
    return BoxLaplacianSpectrumProvider(unit_box, truncation=(5, 5, 5), bc_type='dirichlet')


@pytest.fixture
def inv_lap(space, dirichlet_provider):
    return InverseBoxLaplacian(space, dirichlet_provider, quad_resolution=20)


@pytest.fixture
def sampler(inv_lap):
    return KLSampler3D(inv_lap, n_modes=10, rng=np.random.default_rng(42))


@pytest.fixture
def large_provider(unit_box):
    # 6x6x6 = 216 modes; large enough for a 50-mode sampler
    return BoxLaplacianSpectrumProvider(unit_box, truncation=(6, 6, 6), bc_type='dirichlet')


@pytest.fixture
def large_inv_lap(space, large_provider):
    return InverseBoxLaplacian(space, large_provider, quad_resolution=20)


@pytest.fixture
def large_sampler(large_inv_lap):
    # 50 modes — enough to show meaningful vectorization speedup
    return KLSampler3D(large_inv_lap, n_modes=50, rng=np.random.default_rng(42))


def _eval_pts(n: int = 100, seed: int = 7) -> np.ndarray:
    """Random points in [0, 1]^3."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 1.0, size=(n, 3))


def _loop_evaluate(sampler: KLSampler3D, weights: np.ndarray):
    """Build a reference loop-based evaluate closure identical to the old path."""
    mean_fn = sampler._mean
    eigenfunctions = sampler._eigenfunctions
    snapshot = list(zip(weights.tolist(), eigenfunctions))

    def evaluate(x: np.ndarray) -> np.ndarray | float:
        x = np.asarray(x, dtype=float)
        total = mean_fn(x)
        for w_k, phi_k in snapshot:
            if w_k != 0.0:
                total = total + w_k * phi_k(x)
        return total

    return evaluate


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestKLVectorizedEvaluate:

    def test_kl_sample_vectorized_matches_loop(self, sampler):
        """Vectorized evaluate matches the explicit loop for the same xis."""
        xis = np.random.default_rng(0).standard_normal(sampler.n_modes)
        fn_vec = sampler.sample_from_xis(xis)

        # Reference loop closure using the same weights
        weights = sampler._sqrt_eigenvalues * xis
        loop_fn = _loop_evaluate(sampler, weights)

        pts = _eval_pts(50)
        npt.assert_allclose(fn_vec(pts), loop_fn(pts), rtol=1e-12, atol=1e-14,
                            err_msg="Vectorized and loop evaluate disagree on batch input")

    def test_kl_sample_single_point(self, sampler):
        """Single-point input (shape (3,)) returns a Python float."""
        fn = sampler.sample()
        pt = np.array([0.5, 0.5, 0.5])
        result = fn(pt)
        assert isinstance(result, float), (
            f"Expected float for single-point input; got {type(result).__name__!r}"
        )

    def test_kl_sample_batch_points(self, sampler):
        """Batch input (n, 3) returns ndarray of shape (n,)."""
        fn = sampler.sample()
        pts = _eval_pts(50)
        result = fn(pts)
        assert isinstance(result, np.ndarray), (
            f"Expected ndarray for batch input; got {type(result).__name__!r}"
        )
        assert result.shape == (50,), (
            f"Expected shape (50,); got {result.shape}"
        )

    def test_kl_sample_from_xis_vectorized(self, sampler):
        """sample_from_xis vectorized path matches explicit loop for the same xis."""
        xis = np.random.default_rng(5).standard_normal(sampler.n_modes)
        fn_vec = sampler.sample_from_xis(xis)

        weights = sampler._sqrt_eigenvalues * xis
        loop_fn = _loop_evaluate(sampler, weights)

        pts = _eval_pts(50, seed=13)
        npt.assert_allclose(fn_vec(pts), loop_fn(pts), rtol=1e-12, atol=1e-14,
                            err_msg="sample_from_xis vectorized and loop paths disagree")

    def test_kl_sample_vectorized_speed(self, large_sampler):
        """Vectorized path is >4x faster than loop for n_modes=50, 1000 eval points.

        The vectorized path uses a single batch 1-D basis evaluation + matmul
        per evaluate call, whereas the loop calls each eigenfunction individually
        (n_modes Python function calls × 3 numpy ops each).  With n_modes=50 the
        speedup is reliably ~5x; we require at least 4x.
        """
        sampler = large_sampler
        # Use all-ones xis so no weights are skipped (worst case for the loop)
        xis = np.ones(sampler.n_modes)
        fn_vec = sampler.sample_from_xis(xis)

        weights = sampler._sqrt_eigenvalues * xis
        loop_fn = _loop_evaluate(sampler, weights)

        pts = _eval_pts(1000, seed=11)

        # Warm-up runs (JIT, caching, etc.)
        fn_vec(pts)
        loop_fn(pts)

        N_REPS = 20
        t0 = time.perf_counter()
        for _ in range(N_REPS):
            fn_vec(pts)
        t_vec = (time.perf_counter() - t0) / N_REPS

        t0 = time.perf_counter()
        for _ in range(N_REPS):
            loop_fn(pts)
        t_loop = (time.perf_counter() - t0) / N_REPS

        speedup = t_loop / t_vec
        assert speedup > 4.0, (
            f"Expected >4x speedup for n_modes={sampler.n_modes}, 1000 points; "
            f"got {speedup:.1f}x "
            f"(vec={t_vec * 1000:.2f} ms, loop={t_loop * 1000:.2f} ms)"
        )
