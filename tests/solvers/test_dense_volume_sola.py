"""Tests for DenseVolumeSolaSolver (Phase 3).

Verifies:
1. Gram matrix from DenseVolumeSolaSolver matches generic SOLAOperator3D Gram
2. Resolving coefficients match direct solve via generic Gram matrix
3. Forward map matches SOLAOperator3D forward evaluation
4. SolverRouter recommends DenseVolumeSolaSolver for BoxDomain+GaussianProvider3D+Lebesgue3D

Mathematical background
-----------------------
For m VolumeKernel3D forms {ℓ_i | ℓ_i(f) = ∫ k_i · f dV} on Lebesgue3D:
- Gram matrix  G[i,j] = ⟨k_i, k_j⟩ = ∫ k_i k_j dV ≈ K @ diag(w) @ K^T
- Forward map  d[i] = ℓ_i(f) = K[i,:] · (w * f_vals)
- Resolve      q satisfying G @ q = cross, where cross[i] = ∫ k_i k_target dV
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from pygeoinf import EuclideanSpace

from pygeoinf3d import BoxDomain, Lebesgue3D, Function3D
from pygeoinf3d.providers.function_providers.gaussian import GaussianProvider3D
from pygeoinf3d.spaces.forms import VolumeKernel3D
from pygeoinf3d.operators.sola import SOLAOperator3D
from pygeoinf3d.solvers.dense_volume_sola import DenseVolumeSolaSolver
from pygeoinf3d.solvers.router import SolverRouter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def unit_box():
    return BoxDomain(np.zeros(3), np.ones(3))


@pytest.fixture
def lebesgue_space(unit_box):
    return Lebesgue3D(unit_box)


@pytest.fixture
def gaussian_provider(unit_box):
    """3 Gaussian kernels at different positions, sigma=0.2."""
    centers = np.array([
        [0.2, 0.2, 0.2],
        [0.5, 0.5, 0.5],
        [0.8, 0.8, 0.8],
    ])
    return GaussianProvider3D(unit_box, centers, sigmas=0.2)


@pytest.fixture
def sola_solver(unit_box, gaussian_provider):
    """DenseVolumeSolaSolver with resolution=20 to match domain.integrate_volume default."""
    return DenseVolumeSolaSolver(unit_box, gaussian_provider, resolution=20)


@pytest.fixture
def sola_forms(lebesgue_space, gaussian_provider):
    """Corresponding VolumeKernel3D forms on Lebesgue3D."""
    return [
        VolumeKernel3D(lebesgue_space, gaussian_provider.get_function(i))
        for i in range(gaussian_provider.dim)
    ]


@pytest.fixture
def generic_sola_operator(lebesgue_space, sola_forms):
    """SOLAOperator3D wrapping the same kernels."""
    n = len(sola_forms)
    return SOLAOperator3D(lebesgue_space, EuclideanSpace(n), sola_forms)


@pytest.fixture
def test_function(unit_box):
    """Smooth test function: exp(-3||x - 0.5||²)."""
    def f(x):
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            return float(np.exp(-3.0 * np.sum((x - 0.5) ** 2)))
        return np.exp(-3.0 * np.sum((x - 0.5) ** 2, axis=1))

    return Function3D(unit_box, evaluate_callable=f, name="test_function")


# ---------------------------------------------------------------------------
# Test 1: Gram matrix matches generic SOLAOperator3D
# ---------------------------------------------------------------------------

class TestDenseVolumeSolaGram:

    def test_gram_matches_generic_sola_operator(
        self, sola_solver, generic_sola_operator
    ):
        """DenseVolumeSolaSolver.gram() must equal the Gram matrix of SOLAOperator3D.

        Both use resolution=20 cell-center quadrature for  G[i,j] = ∫ k_i k_j dV,
        so the results must agree to machine precision (<< rtol=1e-8).
        """
        gram_fast = sola_solver.gram()

        # Generic path: (A @ A†).matrix(dense=True, galerkin=True)
        GGT = generic_sola_operator @ generic_sola_operator.adjoint
        gram_generic = GGT.matrix(dense=True, galerkin=True)

        npt.assert_allclose(gram_fast, gram_generic, rtol=1e-8, atol=1e-14)

    def test_gram_is_symmetric(self, sola_solver):
        """Gram matrix must be symmetric."""
        G = sola_solver.gram()
        npt.assert_allclose(G, G.T, atol=1e-14)

    def test_gram_is_positive_definite(self, sola_solver):
        """Gram matrix must have all positive eigenvalues."""
        G = sola_solver.gram()
        eigvals = np.linalg.eigvalsh(G)
        assert np.all(eigvals > 0), f"Non-positive eigenvalues: {eigvals}"


# ---------------------------------------------------------------------------
# Test 2: Resolving coefficients match generic Gram solve
# ---------------------------------------------------------------------------

class TestDenseVolumeSolaResolve:

    def test_resolving_coefficients_match_direct_solve(self, sola_solver):
        """resolve(target_row) must equal np.linalg.solve(Gram, cross).

        Verifies the Cholesky factorisation is correct.
        """
        K = sola_solver._backend.interaction_matrix()
        w = sola_solver._backend._weights

        # Use first kernel's quadrature values as the target
        target_row = K[0, :]
        cross = K @ (w * target_row)

        q_fast = sola_solver.resolve(target_row)
        q_direct = np.linalg.solve(sola_solver.gram(), cross)

        npt.assert_allclose(q_fast, q_direct, rtol=1e-10, atol=1e-14)

    def test_cross_gram_matches_generic_form_evaluation(
        self, sola_solver, sola_forms, gaussian_provider
    ):
        """K @ (w * target_row) must equal [form_i(target_fn) for form_i in forms].

        Confirms that the DenseVolumeSolaSolver's cross-gram computation
        is numerically consistent with VolumeKernel3D.__call__ at resolution=20.
        """
        K = sola_solver._backend.interaction_matrix()
        w = sola_solver._backend._weights

        # Use the last kernel as the target (independent of the first)
        target_idx = len(sola_forms) - 1
        target_fn = gaussian_provider.get_function(target_idx)
        target_row = K[target_idx, :]

        # Fast cross-gram: K @ (w * target_row)
        cross_fast = K @ (w * target_row)

        # Generic cross-gram: form_i(target_fn) = ∫ k_i * target_fn dV
        cross_generic = np.array(
            [form(target_fn) for form in sola_forms], dtype=float
        )

        # Both use resolution=20 quadrature → should match to near machine eps
        npt.assert_allclose(cross_fast, cross_generic, rtol=1e-10, atol=1e-14)

    def test_resolving_satisfies_normal_equations(self, sola_solver):
        """G @ q = cross must hold exactly for the resolved coefficients."""
        K = sola_solver._backend.interaction_matrix()
        w = sola_solver._backend._weights

        target_row = K[1, :]  # use second kernel row as target
        q = sola_solver.resolve(target_row)
        cross = K @ (w * target_row)
        G = sola_solver.gram()

        residual = G @ q - cross
        npt.assert_allclose(residual, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Test 3: Forward map matches SOLAOperator3D
# ---------------------------------------------------------------------------

class TestDenseVolumeSolaForward:

    def test_forward_matches_sola_operator(
        self, sola_solver, generic_sola_operator, test_function
    ):
        """DenseVolumeSolaSolver.forward(f) must equal SOLAOperator3D(f).

        Both compute d[i] = ∫ k_i f dV at resolution=20, so results must agree
        to near machine precision.
        """
        d_fast = sola_solver.forward(test_function)
        d_generic = generic_sola_operator(test_function)

        npt.assert_allclose(d_fast, d_generic, rtol=1e-10, atol=1e-14)

    def test_forward_shape(self, sola_solver, test_function, gaussian_provider):
        """forward(f) returns array of shape (n_kernels,)."""
        d = sola_solver.forward(test_function)
        assert d.shape == (gaussian_provider.dim,)

    def test_forward_is_linear_in_f(self, sola_solver, unit_box):
        """Forward map is linear: forward(a·f + b·g) = a·forward(f) + b·forward(g)."""
        def f_call(x):
            x = np.asarray(x, dtype=float)
            if x.ndim == 1:
                return float(x[0])
            return x[:, 0]

        def g_call(x):
            x = np.asarray(x, dtype=float)
            if x.ndim == 1:
                return float(x[1])
            return x[:, 1]

        f = Function3D(unit_box, evaluate_callable=f_call)
        g = Function3D(unit_box, evaluate_callable=g_call)
        a, b = 3.0, -2.0

        def fg_call(x):
            return a * f_call(x) + b * g_call(x)

        fg = Function3D(unit_box, evaluate_callable=fg_call)

        d_f = sola_solver.forward(f)
        d_g = sola_solver.forward(g)
        d_fg = sola_solver.forward(fg)

        npt.assert_allclose(d_fg, a * d_f + b * d_g, rtol=1e-12, atol=1e-14)


# ---------------------------------------------------------------------------
# Test 4: SolverRouter recommendation
# ---------------------------------------------------------------------------

class TestSolverRouterDenseVolumeSola:

    def test_router_recommends_dense_volume_sola_for_box_gaussian_stack(
        self, unit_box, lebesgue_space, gaussian_provider, sola_forms
    ):
        """SolverRouter.recommend with provider=GaussianProvider3D on Lebesgue3D+BoxDomain
        must recommend DenseVolumeSolaSolver.
        """
        router = SolverRouter()
        rec = router.recommend(lebesgue_space, sola_forms, provider=gaussian_provider)
        assert rec.solver_spec is not None, (
            f"Expected DenseVolumeSolaSolver recommendation, got reason: {rec.reason!r}"
        )
        assert rec.solver_spec.name == "DenseVolumeSolaSolver", (
            f"Expected DenseVolumeSolaSolver, got {rec.solver_spec.name!r}"
        )

    def test_router_still_recommends_sola_without_provider(
        self, lebesgue_space, sola_forms
    ):
        """Without provider kwarg, existing SOLAOperator3D routing must be unchanged."""
        router = SolverRouter()
        rec = router.recommend(lebesgue_space, sola_forms)
        assert rec.solver_spec is not None
        assert rec.solver_spec.name == "SOLAOperator3D"

    def test_router_dense_volume_sola_recommendation_has_reason(
        self, unit_box, lebesgue_space, gaussian_provider, sola_forms
    ):
        """Recommendation must include a non-empty reason string."""
        router = SolverRouter()
        rec = router.recommend(lebesgue_space, sola_forms, provider=gaussian_provider)
        assert isinstance(rec.reason, str) and len(rec.reason) > 0


# ---------------------------------------------------------------------------
# Phase 2 Tests: Push-through identity methods
# ---------------------------------------------------------------------------

class TestPushThroughSolve:
    """Tests for DenseVolumeSolaSolver.solve (push-through identity)."""

    @pytest.fixture
    def noise_std(self):
        return 0.1

    def _old_solve(self, solver, d, noise_std):
        """Reference: data-space Cholesky solve."""
        from scipy.linalg import cho_factor, cho_solve as sp_cho_solve
        K = solver.backend.interaction_matrix()
        w = solver.backend._weights
        N_d = K.shape[0]
        GGT = K @ (w[:, None] * K.T)
        sys = GGT + noise_std**2 * np.eye(N_d)
        return sp_cho_solve(cho_factor(sys), d)

    def test_solve_matches_data_space_cholesky(self, sola_solver, noise_std):
        """push-through solve == data-space Cholesky solve."""
        rng = np.random.default_rng(42)
        N_d = sola_solver.n_kernels
        d = rng.standard_normal(N_d)
        q_new = sola_solver.solve(d, noise_std)
        q_old = self._old_solve(sola_solver, d, noise_std)
        npt.assert_allclose(q_new, q_old, rtol=1e-8)

    def test_solve_multiple_rhs_consistent(self, sola_solver, noise_std):
        """Solve three different RHS vectors, each matches old pipeline."""
        rng = np.random.default_rng(7)
        N_d = sola_solver.n_kernels
        for _ in range(3):
            d = rng.standard_normal(N_d)
            q_new = sola_solver.solve(d, noise_std)
            q_old = self._old_solve(sola_solver, d, noise_std)
            npt.assert_allclose(q_new, q_old, rtol=1e-8)

    def test_solve_caches_model_cho(self, sola_solver, noise_std):
        """Second solve with same noise_std reuses cached Cholesky."""
        rng = np.random.default_rng(99)
        d = rng.standard_normal(sola_solver.n_kernels)
        sola_solver.solve(d, noise_std)
        cho_ref = sola_solver._model_M_cho
        sola_solver.solve(d * 0.5, noise_std)  # same noise_std
        assert sola_solver._model_M_cho is cho_ref  # same object

    def test_solve_invalidates_cache_on_noise_change(self, sola_solver):
        """Changing noise_std rebuilds the cache."""
        rng = np.random.default_rng(55)
        d = rng.standard_normal(sola_solver.n_kernels)
        sola_solver.solve(d, 0.1)
        cho_ref = sola_solver._model_M_cho
        sola_solver.solve(d, 0.2)
        assert sola_solver._model_M_cho is not cho_ref


class TestReconstruct:
    """Tests for DenseVolumeSolaSolver.reconstruct."""

    def test_returns_function3d(self, unit_box, sola_solver, lebesgue_space):
        rng = np.random.default_rng(17)
        d = rng.standard_normal(sola_solver.n_kernels)
        fn = sola_solver.reconstruct(d, 0.1)
        from pygeoinf3d import Function3D
        assert isinstance(fn, Function3D)

    def test_reconstruct_callable_at_single_point(self, unit_box, sola_solver):
        rng = np.random.default_rng(31)
        d = rng.standard_normal(sola_solver.n_kernels)
        fn = sola_solver.reconstruct(d, 0.1)
        val = fn(np.array([[0.5, 0.5, 0.5]]))
        assert np.isfinite(val).all()

    def test_reconstruct_matches_explicit_sum_of_kernels(self, unit_box, gaussian_provider, sola_solver):
        """fn(x) == sum_i q_i k_i(x)."""
        rng = np.random.default_rng(44)
        noise_std = 0.1
        d = rng.standard_normal(sola_solver.n_kernels)
        fn = sola_solver.reconstruct(d, noise_std)
        q  = sola_solver.solve(d, noise_std)
        pts = np.array([[0.2, 0.3, 0.4], [0.7, 0.8, 0.6]])
        K_pts = gaussian_provider.evaluate_all(pts)  # (N_d, 2)
        manual = K_pts.T @ q                          # (2,)
        npt.assert_allclose(fn(pts), manual, rtol=1e-10)


class TestComputeSolaMatrix:
    """Tests for DenseVolumeSolaSolver.compute_sola_matrix."""

    @pytest.fixture
    def target_solver(self, unit_box):
        """Small target solver (5 targets)."""
        centers_T = np.array([
            [0.2, 0.5, 0.5],
            [0.4, 0.5, 0.5],
            [0.5, 0.5, 0.5],
            [0.6, 0.5, 0.5],
            [0.8, 0.5, 0.5],
        ])
        from pygeoinf3d.providers.function_providers.gaussian import GaussianProvider3D
        from pygeoinf3d.solvers.dense_volume_sola import DenseVolumeSolaSolver
        prov_T = GaussianProvider3D(unit_box, centers_T, sigmas=0.2)
        return DenseVolumeSolaSolver(unit_box, prov_T, resolution=(4, 4, 4))

    def test_shape(self, sola_solver, target_solver):
        X = sola_solver.compute_sola_matrix(target_solver, 0.1)
        N_p = target_solver.n_kernels
        N_d = sola_solver.n_kernels
        assert X.shape == (N_p, N_d)

    def test_matches_old_pipeline(self, unit_box, sola_solver, target_solver, lebesgue_space):
        """X @ d should match T_op(G_inv(d)) via old data-space Cholesky pipeline."""
        noise_std = 0.1
        K_G = sola_solver.backend.interaction_matrix()                                # (N_d, N_q)
        # Evaluate target kernels at the forward solver's quadrature pts (same N_q)
        K_T = target_solver.backend._provider.evaluate_all(sola_solver.backend._pts) # (N_p, N_q)
        w   = sola_solver.backend._weights                                            # (N_q,)
        N_d = K_G.shape[0]

        # Old pipeline: data-space Cholesky
        from scipy.linalg import cho_factor, cho_solve as sp_cho
        GGT = K_G @ (w[:, None] * K_G.T)                  # (N_d, N_d)
        sys = GGT + noise_std**2 * np.eye(N_d)
        cho_old = cho_factor(sys)

        # X = K_T @ diag(w) @ K_G^T @ (GGT + sigma^2 I)^{-1}
        TGT = K_T @ (w[:, None] * K_G.T)                  # (N_p, N_d)
        import numpy.linalg as nla
        X_old = TGT @ nla.inv(sys)                         # (N_p, N_d)

        X_new = sola_solver.compute_sola_matrix(target_solver, noise_std)
        npt.assert_allclose(X_new, X_old, rtol=1e-8)

    def test_chunked_equals_full(self, sola_solver, target_solver):
        """Chunked assembly gives same result as full."""
        # Reset cache to ensure fresh computation
        sola_solver._model_M_cho = None
        noise_std = 0.05
        X_full   = sola_solver.compute_sola_matrix(target_solver, noise_std, chunk_size=100)
        sola_solver._model_M_cho = None
        X_chunk1 = sola_solver.compute_sola_matrix(target_solver, noise_std, chunk_size=1)
        npt.assert_allclose(X_chunk1, X_full, rtol=1e-10)


# ---------------------------------------------------------------------------
# Phase 3 Tests: Memory-limited K caching, updated resolve, forward warning
# ---------------------------------------------------------------------------

class TestMemoryBudgetAndLazyK:
    """Tests for memory-budget-controlled K caching."""

    def test_k_cached_by_default(self, unit_box, gaussian_provider):
        """Default solver caches K."""
        solver = DenseVolumeSolaSolver(unit_box, gaussian_provider, resolution=(4, 4, 4))
        _ = solver.backend.interaction_matrix()
        assert solver.backend._K_cache is not None

    def test_k_not_cached_when_budget_exceeded(self, unit_box, gaussian_provider):
        """With memory_budget_bytes=1, K is never cached."""
        solver = DenseVolumeSolaSolver(
            unit_box, gaussian_provider, resolution=(4, 4, 4),
            memory_budget_bytes=1,
        )
        assert not solver._k_fits_in_memory
        assert solver.backend._cache_k is False

    def test_k_cache_none_when_budget_exceeded(self, unit_box, gaussian_provider):
        """When budget exceeded, _K_cache stays None even after interaction_matrix()."""
        solver = DenseVolumeSolaSolver(
            unit_box, gaussian_provider, resolution=(4, 4, 4),
            memory_budget_bytes=1,
        )
        _ = solver.backend.interaction_matrix()
        assert solver.backend._K_cache is None

    def test_solve_works_without_k_cache(self, unit_box, gaussian_provider):
        """solve() still works correctly when K is not cached."""
        solver_cached = DenseVolumeSolaSolver(unit_box, gaussian_provider, resolution=(4, 4, 4))
        solver_streaming = DenseVolumeSolaSolver(
            unit_box, gaussian_provider, resolution=(4, 4, 4), memory_budget_bytes=1
        )
        rng = np.random.default_rng(42)
        d = rng.standard_normal(solver_cached.n_kernels)
        q_c = solver_cached.solve(d, 0.1)
        q_s = solver_streaming.solve(d, 0.1)
        npt.assert_allclose(q_s, q_c, rtol=1e-8)

    def test_no_cache_triggers_warning(self, unit_box, gaussian_provider):
        """interaction_matrix() warns once when cache_k=False."""
        import warnings
        solver = DenseVolumeSolaSolver(
            unit_box, gaussian_provider, resolution=(4, 4, 4),
            memory_budget_bytes=1,
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            solver.backend.interaction_matrix()
            assert any(issubclass(x.category, UserWarning) for x in w)

    def test_no_cache_warning_issued_only_once(self, unit_box, gaussian_provider):
        """The no-cache warning is issued only on the first call."""
        import warnings
        solver = DenseVolumeSolaSolver(
            unit_box, gaussian_provider, resolution=(4, 4, 4),
            memory_budget_bytes=1,
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            solver.backend.interaction_matrix()
            solver.backend.interaction_matrix()
            user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
            assert len(user_warnings) == 1


class TestResolveUpdated:
    """Tests for updated resolve() with noise_std parameter."""

    def test_resolve_with_noise_std_matches_solve(self, unit_box, gaussian_provider):
        """resolve(target_row, noise_std) == solve(K@(w*target), noise_std)."""
        solver = DenseVolumeSolaSolver(unit_box, gaussian_provider, resolution=(4, 4, 4))
        K = solver.backend.interaction_matrix()
        w = solver.backend._weights
        rng = np.random.default_rng(5)
        target = rng.standard_normal(K.shape[1])  # N_q vector
        noise_std = 0.1

        c_t = solver.backend.kernel_matvec(w * target)
        q_solve = solver.solve(c_t, noise_std)
        q_resolve = solver.resolve(target, noise_cov=noise_std)
        npt.assert_allclose(q_resolve, q_solve, rtol=1e-10)

    def test_resolve_legacy_without_noise_warns(self, unit_box, gaussian_provider):
        """resolve(target) without noise_std issues DeprecationWarning."""
        import warnings
        solver = DenseVolumeSolaSolver(unit_box, gaussian_provider, resolution=(4, 4, 4))
        K = solver.backend.interaction_matrix()
        target = K[0, :]  # any row
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            solver.resolve(target)
            assert any(issubclass(x.category, DeprecationWarning) for x in w)

    def test_resolve_no_noise_no_cache_raises(self, unit_box, gaussian_provider):
        """resolve(target) without noise_std raises when K budget exceeded (no _cho)."""
        solver = DenseVolumeSolaSolver(
            unit_box, gaussian_provider, resolution=(4, 4, 4), memory_budget_bytes=1
        )
        K_row = solver.backend._provider.get_function(0)(solver.backend._pts)
        with pytest.raises(RuntimeError, match="noise_std"):
            solver.resolve(K_row)

    def test_resolve_with_noise_works_without_k_cache(self, unit_box, gaussian_provider):
        """resolve(target, noise_std) works even when K is not cached."""
        solver_cached = DenseVolumeSolaSolver(unit_box, gaussian_provider, resolution=(4, 4, 4))
        solver_streaming = DenseVolumeSolaSolver(
            unit_box, gaussian_provider, resolution=(4, 4, 4), memory_budget_bytes=1
        )
        K = solver_cached.backend.interaction_matrix()
        w = solver_cached.backend._weights
        target = K[1, :]
        noise_std = 0.1
        q_cached = solver_cached.resolve(target, noise_cov=noise_std)
        q_stream = solver_streaming.resolve(target, noise_cov=noise_std)
        npt.assert_allclose(q_stream, q_cached, rtol=1e-8)


class TestForwardOperatorWarning:
    """forward_operator() warns when K not cached."""

    def test_warns_when_no_cache(self, unit_box, gaussian_provider, lebesgue_space):
        import warnings
        solver = DenseVolumeSolaSolver(
            unit_box, gaussian_provider, resolution=(4, 4, 4), memory_budget_bytes=1
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            solver.forward_operator(lebesgue_space)
            assert any(
                issubclass(x.category, UserWarning) and "K matrix is not cached" in str(x.message)
                for x in w
            )

    def test_no_warning_when_cached(self, unit_box, gaussian_provider, lebesgue_space):
        import warnings
        solver = DenseVolumeSolaSolver(unit_box, gaussian_provider, resolution=(4, 4, 4))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            solver.forward_operator(lebesgue_space)
            assert not any(
                issubclass(x.category, UserWarning) and "K matrix is not cached" in str(x.message)
                for x in w
            )


# ---------------------------------------------------------------------------
# Phase 4 Tests: Generalised noise covariance (noise_cov parameter)
# ---------------------------------------------------------------------------

class TestGeneralNoiseCov:
    """Tests for generalised C_D support (float, 1-D, and 2-D noise_cov)."""

    # ------------------------------------------------------------------ #
    # Helper: brute-force (C_D + GG*)^{-1} d                              #
    # ------------------------------------------------------------------ #
    def _brute_solve(self, solver, d, C_D_mat):
        K = solver.backend.interaction_matrix()
        w = solver.backend._weights
        GGstar = K @ (w[:, None] * K.T)
        return np.linalg.solve(C_D_mat + GGstar, d)

    # ------------------------------------------------------------------ #
    # solve                                                                #
    # ------------------------------------------------------------------ #

    def test_scalar_solve_matches_brute_force(self, unit_box, gaussian_provider):
        """solve(d, sigma_float) == brute-force (σ²I + GG*)^{-1} d."""
        solver = DenseVolumeSolaSolver(unit_box, gaussian_provider, resolution=(4, 4, 4))
        rng = np.random.default_rng(7)
        N_d = solver.n_kernels
        d = rng.standard_normal(N_d)
        sigma = 0.1
        q = solver.solve(d, sigma)
        C_D = sigma ** 2 * np.eye(N_d)
        q_brute = self._brute_solve(solver, d, C_D)
        npt.assert_allclose(q, q_brute, rtol=1e-8)

    def test_diagonal_uniform_equals_scalar(self, unit_box, gaussian_provider):
        """solve(d, uniform_diag) == solve(d, scalar) when diag is uniform."""
        solver = DenseVolumeSolaSolver(unit_box, gaussian_provider, resolution=(4, 4, 4))
        rng = np.random.default_rng(8)
        d = rng.standard_normal(solver.n_kernels)
        sigma = 0.1
        q_scalar = solver.solve(d, sigma)
        q_diag = solver.solve(d, np.full(solver.n_kernels, sigma))
        npt.assert_allclose(q_diag, q_scalar, rtol=1e-10)

    def test_diagonal_non_uniform_correct(self, unit_box, gaussian_provider):
        """solve(d, per-datum sigmas) matches brute force."""
        solver = DenseVolumeSolaSolver(unit_box, gaussian_provider, resolution=(4, 4, 4))
        rng = np.random.default_rng(10)
        N_d = solver.n_kernels
        d = rng.standard_normal(N_d)
        sigmas = rng.uniform(0.05, 0.3, size=N_d)
        q_diag = solver.solve(d, sigmas)
        C_D_mat = np.diag(sigmas ** 2)
        q_brute = self._brute_solve(solver, d, C_D_mat)
        npt.assert_allclose(q_diag, q_brute, rtol=1e-8)

    def test_dense_scaled_identity_equals_scalar(self, unit_box, gaussian_provider):
        """solve(d, σ²I) == solve(d, σ) for dense C_D = σ²I."""
        solver = DenseVolumeSolaSolver(unit_box, gaussian_provider, resolution=(4, 4, 4))
        rng = np.random.default_rng(9)
        N_d = solver.n_kernels
        d = rng.standard_normal(N_d)
        sigma = 0.1
        C_D = sigma ** 2 * np.eye(N_d)
        q_scalar = solver.solve(d, sigma)
        q_dense = solver.solve(d, C_D)
        npt.assert_allclose(q_dense, q_scalar, rtol=1e-8)

    def test_dense_non_trivial_correct(self, unit_box, gaussian_provider):
        """solve(d, non-trivial SPD C_D) matches brute force."""
        solver = DenseVolumeSolaSolver(unit_box, gaussian_provider, resolution=(4, 4, 4))
        rng = np.random.default_rng(11)
        N_d = solver.n_kernels
        d = rng.standard_normal(N_d)
        A = rng.standard_normal((N_d, N_d))
        C_D = A @ A.T + 0.5 * np.eye(N_d)
        q_dense = solver.solve(d, C_D)
        q_brute = self._brute_solve(solver, d, C_D)
        npt.assert_allclose(q_dense, q_brute, rtol=1e-6)

    def test_dense_requires_k_cached(self, unit_box, gaussian_provider):
        """Dense C_D raises RuntimeError when K not cached."""
        solver = DenseVolumeSolaSolver(
            unit_box, gaussian_provider, resolution=(4, 4, 4), memory_budget_bytes=1
        )
        N_d = solver.n_kernels
        C_D = np.eye(N_d)
        d = np.ones(N_d)
        with pytest.raises(RuntimeError, match="memory_budget_bytes"):
            solver.solve(d, C_D)

    # ------------------------------------------------------------------ #
    # compute_sola_matrix                                                  #
    # ------------------------------------------------------------------ #

    @pytest.fixture
    def small_target_solver(self, unit_box):
        centers = np.array([[0.3, 0.5, 0.5], [0.7, 0.5, 0.5]])
        from pygeoinf3d.providers.function_providers.gaussian import GaussianProvider3D
        prov = GaussianProvider3D(unit_box, centers, sigmas=0.2)
        return DenseVolumeSolaSolver(unit_box, prov, resolution=(4, 4, 4))

    def test_compute_sola_matrix_diagonal_uniform_equals_scalar(
        self, unit_box, gaussian_provider, small_target_solver
    ):
        """compute_sola_matrix uniform diagonal == scalar."""
        solver = DenseVolumeSolaSolver(unit_box, gaussian_provider, resolution=(4, 4, 4))
        sigma = 0.1
        X_scalar = solver.compute_sola_matrix(small_target_solver, sigma)
        X_diag = solver.compute_sola_matrix(
            small_target_solver, np.full(solver.n_kernels, sigma)
        )
        npt.assert_allclose(X_diag, X_scalar, rtol=1e-10)

    def test_compute_sola_matrix_dense_scaled_identity_equals_scalar(
        self, unit_box, gaussian_provider, small_target_solver
    ):
        """compute_sola_matrix with σ²I == scalar."""
        solver = DenseVolumeSolaSolver(unit_box, gaussian_provider, resolution=(4, 4, 4))
        sigma = 0.1
        N_d = solver.n_kernels
        C_D = sigma ** 2 * np.eye(N_d)
        X_scalar = solver.compute_sola_matrix(small_target_solver, sigma)
        X_dense = solver.compute_sola_matrix(small_target_solver, C_D)
        npt.assert_allclose(X_dense, X_scalar, rtol=1e-8)

    # ------------------------------------------------------------------ #
    # Backward compatibility shim                                          #
    # ------------------------------------------------------------------ #

    def test_noise_std_kwarg_deprecated_in_solve(self, unit_box, gaussian_provider):
        """solve(d, noise_std=0.1) issues DeprecationWarning."""
        import warnings
        solver = DenseVolumeSolaSolver(unit_box, gaussian_provider, resolution=(4, 4, 4))
        d = np.ones(solver.n_kernels)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            solver.solve(d, noise_std=0.1)
            assert any(
                issubclass(x.category, DeprecationWarning) and "noise_std" in str(x.message)
                for x in w
            )

    def test_noise_std_kwarg_gives_same_result(self, unit_box, gaussian_provider):
        """solve(d, noise_std=σ) == solve(d, σ) (same float value, same semantics)."""
        import warnings
        solver = DenseVolumeSolaSolver(unit_box, gaussian_provider, resolution=(4, 4, 4))
        rng = np.random.default_rng(42)
        d = rng.standard_normal(solver.n_kernels)
        q_new = solver.solve(d, 0.1)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            q_old = solver.solve(d, noise_std=0.1)
        npt.assert_allclose(q_new, q_old, rtol=1e-12)
