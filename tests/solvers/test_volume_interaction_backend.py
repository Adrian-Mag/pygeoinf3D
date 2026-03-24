"""Tests for VolumeInteractionBackend and FunctionProvider3D.evaluate_all.

Verifies:
- FunctionProvider3D.evaluate_all default loop fallback
- GaussianProvider3D.evaluate_all vectorized matches loop
- VolumeInteractionBackend.interaction_matrix matches kernel-by-kernel quadrature
- VolumeInteractionBackend.gram matches direct weighted sum
- Fallback for providers without a fast evaluate_all gives correct results
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from pygeoinf3d import BoxDomain, Function3D
from pygeoinf3d.providers.function_providers.gaussian import GaussianProvider3D
from pygeoinf3d.providers.base import FunctionProvider3D
from pygeoinf3d.solvers.volume_interactions import VolumeInteractionBackend


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def unit_box():
    return BoxDomain(np.zeros(3), np.ones(3))


@pytest.fixture
def small_gaussian_provider(unit_box):
    centers = np.array([
        [0.25, 0.25, 0.25],
        [0.50, 0.50, 0.50],
        [0.75, 0.75, 0.75],
    ])
    return GaussianProvider3D(unit_box, centers, sigmas=0.15)


@pytest.fixture
def small_backend(unit_box, small_gaussian_provider):
    return VolumeInteractionBackend(unit_box, small_gaussian_provider, resolution=(4, 4, 4))


# ---------------------------------------------------------------------------
# Mock provider that has NO vectorized evaluate_all — exercises loop fallback
# ---------------------------------------------------------------------------

class _LinearMockProvider(FunctionProvider3D):
    """Simple provider whose i-th function is f_i(x) = x[:,0]*i + x[:,1]."""

    def __init__(self, domain, n: int):
        self._domain = domain
        self._n = n

    @property
    def dim(self) -> int:
        return self._n

    def get_function(self, i: int) -> Function3D:
        slope = float(i) + 1.0

        def _eval(x, _slope=slope):
            x = np.asarray(x, dtype=float)
            if x.ndim == 1:
                return float(_slope * x[0] + x[1])
            return _slope * x[:, 0] + x[:, 1]

        return Function3D(self._domain, evaluate_callable=_eval, name=f"linear_{i}")


# ---------------------------------------------------------------------------
# Test 1: GaussianProvider3D.evaluate_all matches loop
# ---------------------------------------------------------------------------

class TestEvaluateAll:

    def test_gaussian_provider_evaluate_all_matches_get_function_loop(
        self, small_gaussian_provider
    ):
        """evaluate_all(pts) should agree with looping over get_function(i)(pts)."""
        rng = np.random.default_rng(0)
        pts = rng.uniform(0.0, 1.0, size=(50, 3))

        result = small_gaussian_provider.evaluate_all(pts)

        n = small_gaussian_provider.dim
        assert result.shape == (n, 50), (
            f"Expected shape ({n}, 50), got {result.shape}"
        )

        for i in range(n):
            fi = small_gaussian_provider.get_function(i)
            expected_row = fi(pts)
            npt.assert_allclose(
                result[i, :], expected_row, rtol=1e-10,
                err_msg=f"Row {i} mismatch between evaluate_all and loop"
            )


# ---------------------------------------------------------------------------
# Test 2: interaction_matrix matches kernel-by-kernel quadrature
# ---------------------------------------------------------------------------

class TestVolumeInteractionBackend:

    def test_interaction_matrix_matches_kernel_by_kernel_quadrature(
        self, unit_box, small_gaussian_provider, small_backend
    ):
        """K[i, :] from backend should match manual evaluation at quad points."""
        pts = unit_box.structured_mesh((4, 4, 4), location="cell_centers")
        n_pts = pts.shape[0]

        K = small_backend.interaction_matrix()
        n_kernels = small_gaussian_provider.dim

        assert K.shape == (n_kernels, n_pts), (
            f"Expected shape ({n_kernels}, {n_pts}), got {K.shape}"
        )

        for i in range(n_kernels):
            fi = small_gaussian_provider.get_function(i)
            manual_row = fi(pts)
            npt.assert_allclose(
                K[i, :], manual_row, rtol=1e-10,
                err_msg=f"K[{i}, :] mismatch between backend and manual evaluation"
            )

    def test_gram_matches_direct_integration(
        self, unit_box, small_backend
    ):
        """G[i,j] = sum(K[i,:] * K[j,:] * w) must hold for all (i,j)."""
        K = small_backend.interaction_matrix()
        w = unit_box.volume_weights((4, 4, 4), location="cell_centers")
        G = small_backend.gram()

        n_kernels = K.shape[0]
        assert G.shape == (n_kernels, n_kernels)

        # G = K @ diag(w) @ K.T
        G_expected = K @ (w[:, None] * K.T)
        npt.assert_allclose(G, G_expected, rtol=1e-10,
                            err_msg="gram() does not match K @ diag(w) @ K.T")

    def test_volume_interaction_backend_falls_back_for_provider_without_evaluate_all(
        self, unit_box
    ):
        """A provider with only the loop-fallback evaluate_all should still give correct results."""
        mock_provider = _LinearMockProvider(unit_box, n=3)
        backend = VolumeInteractionBackend(unit_box, mock_provider, resolution=(4, 4, 4))

        pts = unit_box.structured_mesh((4, 4, 4), location="cell_centers")
        K = backend.interaction_matrix()

        assert K.shape == (3, pts.shape[0])

        # Verify each row matches manual loop
        for i in range(3):
            fi = mock_provider.get_function(i)
            manual_row = fi(pts)
            npt.assert_allclose(
                K[i, :], manual_row, rtol=1e-10,
                err_msg=f"Fallback K[{i}, :] mismatch"
            )

    def test_cross_gram_matches_direct(self, unit_box, small_gaussian_provider):
        """cross_gram(other) should equal K_self @ diag(w) @ K_other.T."""
        backend_a = VolumeInteractionBackend(
            unit_box, small_gaussian_provider, resolution=(4, 4, 4)
        )
        # Build a second backend with a different provider on the same domain
        centers2 = np.array([[0.3, 0.3, 0.3], [0.6, 0.6, 0.6]])
        provider2 = GaussianProvider3D(unit_box, centers2, sigmas=0.15)
        backend_b = VolumeInteractionBackend(
            unit_box, provider2, resolution=(4, 4, 4)
        )

        G_cross = backend_a.cross_gram(backend_b)
        K_a = backend_a.interaction_matrix()
        K_b = backend_b.interaction_matrix()
        w = unit_box.volume_weights((4, 4, 4), location="cell_centers")
        G_expected = K_a @ (w[:, np.newaxis] * K_b.T)

        assert G_cross.shape == (K_a.shape[0], K_b.shape[0])
        npt.assert_allclose(G_cross, G_expected, rtol=1e-10,
                            err_msg="cross_gram does not match K_a @ diag(w) @ K_b.T")

    def test_cross_gram_raises_for_incompatible_backends(
        self, unit_box, small_gaussian_provider
    ):
        """cross_gram must raise ValueError when resolutions differ."""
        backend_a = VolumeInteractionBackend(
            unit_box, small_gaussian_provider, resolution=(4, 4, 4)
        )
        backend_coarse = VolumeInteractionBackend(
            unit_box, small_gaussian_provider, resolution=(6, 6, 6)
        )
        with pytest.raises(ValueError):
            backend_a.cross_gram(backend_coarse)


# ---------------------------------------------------------------------------
# Test 3: evaluate_batch
# ---------------------------------------------------------------------------

class TestEvaluateBatch:
    """Tests for FunctionProvider3D.evaluate_batch."""

    def test_gaussian_evaluate_batch_matches_evaluate_all(
        self, unit_box, small_gaussian_provider, small_backend
    ):
        pts = small_backend._pts
        K_all = small_gaussian_provider.evaluate_all(pts)
        K_batch = small_gaussian_provider.evaluate_batch(
            0, small_gaussian_provider.dim, pts
        )
        npt.assert_allclose(K_batch, K_all, rtol=1e-12)

    def test_gaussian_evaluate_batch_slice_matches_row(
        self, unit_box, small_gaussian_provider, small_backend
    ):
        pts = small_backend._pts
        K_all = small_gaussian_provider.evaluate_all(pts)
        K_batch = small_gaussian_provider.evaluate_batch(1, 2, pts)
        npt.assert_allclose(K_batch, K_all[1:2], rtol=1e-12)

    def test_base_class_evaluate_batch_fallback(self, unit_box):
        """The _LinearMockProvider uses the default loop-based evaluate_all."""
        mock_provider = _LinearMockProvider(unit_box, n=3)
        pts = unit_box.structured_mesh((4, 4, 4), location="cell_centers")
        K_all = mock_provider.evaluate_all(pts)
        K_batch = mock_provider.evaluate_batch(0, mock_provider.dim, pts)
        npt.assert_allclose(K_batch, K_all, rtol=1e-12)


# ---------------------------------------------------------------------------
# Test 4: build_model_gram
# ---------------------------------------------------------------------------

class TestBuildModelGram:
    """Tests for VolumeInteractionBackend.build_model_gram."""

    def test_matches_explicit_gram(self, small_backend):
        """M_CD = I + K̃^T K̃ / σ² should match explicit computation."""
        noise_std = 0.1
        K = small_backend.interaction_matrix()
        w = small_backend._weights
        sqrt_w = np.sqrt(w)
        Kw = K * sqrt_w[None, :]  # (N_d, N_q)
        N_q = len(w)
        M_expected = np.eye(N_q) + (Kw.T @ Kw) / noise_std ** 2
        M, cho, noise_model = small_backend.build_model_gram(noise_std)
        npt.assert_allclose(M, M_expected, rtol=1e-10)
        # noise_model should give correct C_D^{-1}: d / σ²
        d_test = np.ones(small_backend._provider.dim)
        npt.assert_allclose(noise_model.apply_inv(d_test), d_test / noise_std ** 2, rtol=1e-12)

    def test_chunked_equals_full(self, small_backend):
        """chunk_size=1, chunk_size=2, chunk_size=N_d all agree."""
        noise_std = 0.05
        M_full, _, _ = small_backend.build_model_gram(noise_std, chunk_size=100)
        M_chunk1, _, _ = small_backend.build_model_gram(noise_std, chunk_size=1)
        M_chunk2, _, _ = small_backend.build_model_gram(noise_std, chunk_size=2)
        npt.assert_allclose(M_chunk1, M_full, rtol=1e-10)
        npt.assert_allclose(M_chunk2, M_full, rtol=1e-10)


# ---------------------------------------------------------------------------
# Test 5: normal_vector and kernel_matvec
# ---------------------------------------------------------------------------

class TestNormalVectorAndKernelMatvec:
    """Tests for normal_vector and kernel_matvec."""

    def test_normal_vector_matches_explicit(self, small_backend):
        """K̃^T d = sqrt(w) * K^T d."""
        K = small_backend.interaction_matrix()
        w = small_backend._weights
        sqrt_w = np.sqrt(w)
        rng = np.random.default_rng(42)
        d = rng.standard_normal(small_backend._provider.dim)
        expected = sqrt_w * (K.T @ d)
        result = small_backend.normal_vector(d)
        npt.assert_allclose(result, expected, rtol=1e-10)

    def test_normal_vector_chunked_equals_cached(self, small_backend):
        """Chunked path (after clearing K cache) gives same result."""
        K = small_backend.interaction_matrix()   # warm up cache
        rng = np.random.default_rng(7)
        d = rng.standard_normal(small_backend._provider.dim)
        result_cached = small_backend.normal_vector(d)
        # Force chunked path by temporarily hiding cache
        original_cache = small_backend._K_cache
        small_backend._K_cache = None
        result_chunked = small_backend.normal_vector(d, chunk_size=1)
        small_backend._K_cache = original_cache
        npt.assert_allclose(result_chunked, result_cached, rtol=1e-10)

    def test_kernel_matvec_matches_explicit(self, small_backend):
        """K v should match explicit K @ v."""
        K = small_backend.interaction_matrix()
        rng = np.random.default_rng(99)
        v = rng.standard_normal(K.shape[1])
        expected = K @ v
        result = small_backend.kernel_matvec(v)
        npt.assert_allclose(result, expected, rtol=1e-10)

    def test_kernel_matvec_chunked_equals_cached(self, small_backend):
        """Chunked path gives same result."""
        K = small_backend.interaction_matrix()
        rng = np.random.default_rng(13)
        v = rng.standard_normal(K.shape[1])
        result_cached = small_backend.kernel_matvec(v)
        original_cache = small_backend._K_cache
        small_backend._K_cache = None
        result_chunked = small_backend.kernel_matvec(v, chunk_size=1)
        small_backend._K_cache = original_cache
        npt.assert_allclose(result_chunked, result_cached, rtol=1e-10)


# ---------------------------------------------------------------------------
# Test 6: build_model_gram — generalised noise covariance
# ---------------------------------------------------------------------------

class TestBuildModelGramGeneralNoise:
    """Tests for build_model_gram with scalar, 1-D, and 2-D noise_cov."""

    def test_scalar_m_cd_formula(self, small_backend):
        """M_CD = I + K̃^T K̃ / σ² for scalar noise_cov = σ (std dev)."""
        sigma = 0.1
        K = small_backend.interaction_matrix()
        w = small_backend._weights
        sqrt_w = np.sqrt(w)
        Kw = K * sqrt_w[None, :]
        N_q = len(w)
        M_expected = np.eye(N_q) + (Kw.T @ Kw) / sigma ** 2
        M, cho, noise_model = small_backend.build_model_gram(sigma)
        npt.assert_allclose(M, M_expected, rtol=1e-10)

    def test_scalar_noise_model_applies_inv_correctly(self, small_backend):
        """noise_model.apply_inv(d) == d / σ² for scalar noise_cov."""
        sigma = 0.1
        rng = np.random.default_rng(42)
        d = rng.standard_normal(small_backend._provider.dim)
        _, _, noise_model = small_backend.build_model_gram(sigma)
        npt.assert_allclose(noise_model.apply_inv(d), d / sigma ** 2, rtol=1e-12)

    def test_diagonal_m_cd_matches_scalar_when_uniform(self, small_backend):
        """build_model_gram with uniform 1-D noise == scalar version."""
        sigma = 0.1
        N_d = small_backend._provider.dim
        M_scalar, _, _ = small_backend.build_model_gram(sigma)
        M_diag, _, _ = small_backend.build_model_gram(np.full(N_d, sigma))
        npt.assert_allclose(M_diag, M_scalar, rtol=1e-10)

    def test_diagonal_wrong_length_raises(self, small_backend):
        """1-D noise_cov with wrong length raises ValueError."""
        with pytest.raises(ValueError, match="N_d"):
            small_backend.build_model_gram(np.array([0.1, 0.2]))

    def test_diagonal_noise_model_applies_inv_correctly(self, small_backend):
        """noise_model.apply_inv(d) == d / σ_i² for per-datum noise."""
        N_d = small_backend._provider.dim
        rng = np.random.default_rng(7)
        sigmas = rng.uniform(0.05, 0.3, size=N_d)
        d = rng.standard_normal(N_d)
        _, _, noise_model = small_backend.build_model_gram(sigmas)
        npt.assert_allclose(noise_model.apply_inv(d), d / sigmas ** 2, rtol=1e-12)

    def test_dense_m_cd_matches_scalar_when_cd_is_scaled_identity(self, small_backend):
        """build_model_gram with σ²I (dense) == scalar version."""
        sigma = 0.1
        N_d = small_backend._provider.dim
        C_D = sigma ** 2 * np.eye(N_d)
        # Ensure K is cached
        _ = small_backend.interaction_matrix()
        M_scalar, _, _ = small_backend.build_model_gram(sigma)
        M_dense, _, _ = small_backend.build_model_gram(C_D)
        npt.assert_allclose(M_dense, M_scalar, rtol=1e-8)

    def test_dense_requires_k_cached(self, unit_box, small_gaussian_provider):
        """Dense noise_cov raises RuntimeError when K not cached."""
        backend = VolumeInteractionBackend(
            unit_box, small_gaussian_provider, resolution=(4, 4, 4), cache_k=False
        )
        N_d = small_gaussian_provider.dim
        C_D = np.eye(N_d)
        with pytest.raises(RuntimeError, match="cached"):
            backend.build_model_gram(C_D)

    def test_scalar_chunked_equals_full(self, small_backend):
        """Chunked == full for scalar noise_cov."""
        sigma = 0.05
        M_full, _, _ = small_backend.build_model_gram(sigma, chunk_size=100)
        M_c1, _, _ = small_backend.build_model_gram(sigma, chunk_size=1)
        npt.assert_allclose(M_c1, M_full, rtol=1e-10)

    def test_diagonal_chunked_equals_full(self, small_backend):
        """Chunked == full for 1-D noise_cov."""
        N_d = small_backend._provider.dim
        rng = np.random.default_rng(99)
        sigmas = rng.uniform(0.05, 0.3, size=N_d)
        M_full, _, _ = small_backend.build_model_gram(sigmas, chunk_size=100)
        M_c1, _, _ = small_backend.build_model_gram(sigmas, chunk_size=1)
        npt.assert_allclose(M_c1, M_full, rtol=1e-10)
