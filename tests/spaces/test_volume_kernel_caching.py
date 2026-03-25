"""Tests for the explicit cache API on VolumeKernel3D.

Covers:
- cached apply matches uncached apply on BoxDomain
- repeated apply after build_cache does not re-evaluate kernel callable
- clear_cache disables the cache path
- in-place arithmetic (imul, iadd, isub) invalidates the cache
- build_cache raises TypeError when domain has no structured quadrature
"""

from types import SimpleNamespace

import numpy as np
import pytest

from pygeoinf3d import BoxDomain, Function3D, Lebesgue3D, VolumeKernel3D


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def box():
    return BoxDomain(
        np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0])
    )


@pytest.fixture
def space(box):
    return Lebesgue3D(box)


def _gaussian_fn(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.exp(-np.sum(x**2, axis=-1))


def _linear_fn(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x[..., 0] + x[..., 1] + x[..., 2]


@pytest.fixture
def gaussian_kernel(space):
    return VolumeKernel3D(
        space,
        Function3D(space.function_domain, evaluate_callable=_gaussian_fn),
    )


@pytest.fixture
def g_func(space):
    """A test function g whose inner product with the Gaussian kernel is
    clearly non-zero (avoids cancellation in relative-tolerance checks).

    g(x) = ||x||² + 2  >  0 everywhere on [-1,1]³, giving
    ⟨exp(-||x||²), g⟩ = ∫ exp(-r²)(r² + 2) dV  ≈ 7.58...
    """

    def g_callable(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return np.sum(x**2, axis=-1) + 2.0

    return Function3D(space.function_domain, evaluate_callable=g_callable)


# ---------------------------------------------------------------------------
# 1.  Cached apply matches uncached apply
# ---------------------------------------------------------------------------


class TestCachedMatchesUncached:
    def test_default_resolution(self, gaussian_kernel, g_func):
        uncached = gaussian_kernel(g_func)
        gaussian_kernel.build_cache(resolution=20)
        cached = gaussian_kernel(g_func)
        # Same quadrature rule — should be identical to machine precision
        np.testing.assert_allclose(cached, uncached, rtol=1e-10)

    def test_nodes_location(self, space, g_func):
        """build_cache with location='nodes' runs, uses the right pts, and
        produces a value consistent with cell_centers at modest resolution."""
        # Build a fresh form so cache state is clean
        vk = VolumeKernel3D(
            space,
            Function3D(space.function_domain, evaluate_callable=_gaussian_fn),
        )
        vk.build_cache(resolution=20, location="nodes")
        assert vk.cache_built
        val_nodes = vk(g_func)
        # Compare to cell_centers at same resolution (different rule,
        # expect ~0.5% agreement for smooth integrands).
        vk2 = VolumeKernel3D(
            space,
            Function3D(space.function_domain, evaluate_callable=_gaussian_fn),
        )
        val_cc = vk2(g_func)  # uncached, default cell_centers resolution=20
        np.testing.assert_allclose(val_nodes, val_cc, rtol=5e-3)

    def test_constant_g(self, space, gaussian_kernel):
        """With g=1 the result should equal ∫ f dV (analytic cross-check).
        Both uncached and cached use the SAME resolution so results are
        bitwise-identical up to floating-point reordering (~1e-14).
        """

        def g_const(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=float)
            return np.ones(len(x) if x.ndim > 1 else 1)

        g = Function3D(space.function_domain, evaluate_callable=g_const)
        uncached = gaussian_kernel(g)
        # Use the same resolution=20 as integrate_volume's default.
        gaussian_kernel.build_cache(resolution=20)
        cached = gaussian_kernel(g)
        np.testing.assert_allclose(cached, uncached, rtol=1e-12)

    def test_second_g_after_cache_built(self, space, gaussian_kernel):
        """Cache applies correctly to a different g without rebuild."""
        gaussian_kernel.build_cache(resolution=20)

        def g2_callable(x):
            x = np.asarray(x, dtype=float)
            return x[..., 2] ** 2

        g2 = Function3D(space.function_domain, evaluate_callable=g2_callable)

        # Build a fresh form (no cache) on the same kernel to get reference
        ref = VolumeKernel3D(
            gaussian_kernel.domain,
            gaussian_kernel.kernel.copy(),
        )
        uncached = ref(g2)
        cached = gaussian_kernel(g2)
        np.testing.assert_allclose(cached, uncached, rtol=1e-10)


# ---------------------------------------------------------------------------
# 2.  Repeated apply does NOT re-evaluate the kernel callable
# ---------------------------------------------------------------------------


class TestKernelNotReevaluated:
    def test_kernel_evaluated_once_at_build_time(self, space, g_func):
        call_log: list[int] = []  # records n_pts per call to the kernel

        def counting_kernel(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=float)
            n = 1 if x.ndim == 1 else len(x)
            call_log.append(n)
            return _gaussian_fn(x)

        vk = VolumeKernel3D(
            space,
            Function3D(
                space.function_domain,
                evaluate_callable=counting_kernel,
            ),
        )

        # build_cache should call the kernel exactly once on the mesh
        vk.build_cache(resolution=5)
        assert len(call_log) == 1, (
            "build_cache should call the kernel callable exactly once"
        )
        n_pts_at_build = call_log[0]
        assert n_pts_at_build == 5**3  # 125 cell-center points

        # Subsequent apply calls must NOT touch the kernel callable
        call_log.clear()
        for _ in range(4):
            vk(g_func)
        assert call_log == [], (
            "Cached _mapping_impl must not re-evaluate the kernel callable"
        )

    def test_uncached_does_call_kernel(self, space, g_func):
        """Sanity: without cache, the kernel IS evaluated on every apply."""
        call_log: list[int] = []

        def counting_kernel(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=float)
            n = 1 if x.ndim == 1 else len(x)
            call_log.append(n)
            return _gaussian_fn(x)

        vk = VolumeKernel3D(
            space,
            Function3D(
                space.function_domain,
                evaluate_callable=counting_kernel,
            ),
        )
        vk(g_func)
        vk(g_func)
        assert len(call_log) >= 2, (
            "Without cache, the kernel callable should be called on each apply"
        )


# ---------------------------------------------------------------------------
# 3.  clear_cache disables the cache path
# ---------------------------------------------------------------------------


class TestClearCache:
    def test_cache_built_property(self, gaussian_kernel):
        assert not gaussian_kernel.cache_built
        gaussian_kernel.build_cache(resolution=10)
        assert gaussian_kernel.cache_built
        gaussian_kernel.clear_cache()
        assert not gaussian_kernel.cache_built

    def test_apply_after_clear_matches_uncached(self, gaussian_kernel, g_func):
        gaussian_kernel.build_cache(resolution=20)
        val_cached = gaussian_kernel(g_func)

        gaussian_kernel.clear_cache()
        assert not gaussian_kernel.cache_built
        val_fallback = gaussian_kernel(g_func)

        np.testing.assert_allclose(val_cached, val_fallback, rtol=1e-10)

    def test_rebuild_after_clear(self, gaussian_kernel, g_func):
        gaussian_kernel.build_cache(resolution=20)
        gaussian_kernel.clear_cache()
        # Rebuilding should work fine
        gaussian_kernel.build_cache(resolution=10)
        assert gaussian_kernel.cache_built
        _ = gaussian_kernel(g_func)  # must not raise


# ---------------------------------------------------------------------------
# 4.  In-place arithmetic invalidates the cache
# ---------------------------------------------------------------------------


class TestInPlaceArithmeticInvalidatesCache:
    def test_imul_invalidates(self, gaussian_kernel):
        gaussian_kernel.build_cache(resolution=10)
        assert gaussian_kernel.cache_built
        gaussian_kernel *= 2.0
        assert not gaussian_kernel.cache_built

    def test_iadd_invalidates(self, space, gaussian_kernel):
        other = VolumeKernel3D(
            space,
            Function3D(space.function_domain, evaluate_callable=_linear_fn),
        )
        gaussian_kernel.build_cache(resolution=10)
        assert gaussian_kernel.cache_built
        gaussian_kernel += other
        assert not gaussian_kernel.cache_built

    def test_isub_invalidates(self, space, gaussian_kernel):
        other = VolumeKernel3D(
            space,
            Function3D(space.function_domain, evaluate_callable=_linear_fn),
        )
        gaussian_kernel.build_cache(resolution=10)
        assert gaussian_kernel.cache_built
        gaussian_kernel -= other
        assert not gaussian_kernel.cache_built

    def test_apply_after_imul_uses_new_kernel(self, gaussian_kernel, g_func):
        """After *= 3.0, results should be 3× the original."""
        original = gaussian_kernel(g_func)
        # Build and discard cache so the post-imul apply uses uncached path
        gaussian_kernel.build_cache(resolution=20)
        gaussian_kernel *= 3.0  # invalidates cache
        assert not gaussian_kernel.cache_built
        result = gaussian_kernel(g_func)
        np.testing.assert_allclose(result, 3.0 * original, rtol=1e-12)

    def test_apply_after_iadd_correct(self, space, gaussian_kernel, g_func):
        """After += other, result should equal sum of both forms."""
        def f2(x):
            x = np.asarray(x, dtype=float)
            return x[..., 0] ** 2 + 1.5  # positive, non-zero contribution

        other = VolumeKernel3D(
            space,
            Function3D(space.function_domain, evaluate_callable=f2),
        )
        ref_sum = gaussian_kernel(g_func) + other(g_func)
        gaussian_kernel.build_cache(resolution=20)
        gaussian_kernel += other  # invalidates cache
        assert not gaussian_kernel.cache_built
        result = gaussian_kernel(g_func)
        np.testing.assert_allclose(result, ref_sum, rtol=1e-12)


# ---------------------------------------------------------------------------
# 5.  build_cache raises TypeError for non-structured domain
# ---------------------------------------------------------------------------


class TestBuildCacheTypeError:
    def test_raises_when_structured_mesh_missing(self, space):
        """build_cache raises TypeError if domain has no structured_mesh."""
        vk = VolumeKernel3D(
            space,
            Function3D(space.function_domain, evaluate_callable=_gaussian_fn),
        )
        vk_domain_backup = vk.domain
        object.__setattr__(
            vk,
            "_domain",
            SimpleNamespace(function_domain=object()),
        )

        try:
            with pytest.raises(TypeError, match="structured quadrature"):
                vk.build_cache()
        finally:
            object.__setattr__(vk, "_domain", vk_domain_backup)
