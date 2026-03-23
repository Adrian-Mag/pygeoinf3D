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
