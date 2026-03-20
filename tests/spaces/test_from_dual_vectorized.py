"""Tests for vectorized Branch 3 of Sobolev3D.from_dual.

Phase 4 of vectorize-hot-paths-plan.md.

Covers:
- Fast path (PathIntegralFunctional + BoxLaplacianSpectrumProvider):
  vectorized analysis AND synthesis steps.
- Generic form slow-analysis path (custom LinearForm subclass + BLSP provider):
  loop analysis, vectorized synthesis.
- Fallback loop path (PathIntegralFunctional with non-BLSP SpectralOperator3D):
  loop for everything — correct results, no crash.
- Single-point evaluation returns scalar.
- Batch-point evaluation returns array.
- Branch 1 (VolumeKernel3D) regression — still works correctly.
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from pygeoinf import LinearForm
from pygeoinf3d.core.functions import Function3D
from pygeoinf3d.spaces.lebesgue import Lebesgue3D
from pygeoinf3d.spaces.sobolev import Sobolev3D


# ---------------------------------------------------------------------------
# Helper: a custom LinearForm subclass (mirrors VolumeKernel3D / PIF pattern)
# ---------------------------------------------------------------------------

class _PointEvalLinearForm(LinearForm):
    """A LinearForm that evaluates f at a fixed 3D point.

    Follows the exact same pattern as VolumeKernel3D and PathIntegralFunctional:
    - store state BEFORE calling super().__init__
    - override _mapping_impl

    Because LinearForm.__init__ resolves self._mapping_impl dynamically through
    Python MRO (as documented in forms.py), the overridden method is captured.
    """

    def __init__(self, space, center: np.ndarray) -> None:
        self._eval_center = np.asarray(center, dtype=float).reshape(1, 3)
        # Pass components=np.zeros(0) to bypass component-enumeration logic.
        super().__init__(space, components=np.zeros(0))

    def _mapping_impl(self, f: Function3D) -> float:
        return float(f(self._eval_center)[0])


# ---------------------------------------------------------------------------
# Helper: a minimal SpectralOperator3D with no _provider (for fallback tests)
# ---------------------------------------------------------------------------

def _make_minimal_spectral_no_provider(space, box, n_modes=2):
    """Return a SpectralOperator3D backed by simple eigenfunctions, no _provider."""
    from pygeoinf3d.operators.base import SpectralOperator3D

    _evals = [float(k + 1) for k in range(n_modes)]
    _efuncs = [
        Function3D(
            box,
            evaluate_callable=(
                (lambda k_val: lambda x: (
                    np.full(len(x), float(k_val + 1)) if x.ndim == 2
                    else float(k_val + 1)
                ))(k)
            ),
        )
        for k in range(n_modes)
    ]

    class _MinimalSpectral(SpectralOperator3D):
        def __init__(self, _space):
            def _apply(f: Function3D) -> Function3D:
                return f
            super().__init__(_space, _space, _apply)

        def get_eigenvalue(self, index: int) -> float:
            return _evals[index]

        def get_eigenfunction(self, index: int) -> Function3D:
            return _efuncs[index]

        @property
        def n_modes(self) -> int:
            return n_modes

    op = _MinimalSpectral(space)
    assert not hasattr(op, '_provider'), "should have no _provider"
    return op


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def box():
    from pygeoinf3d.geometries.box.domain import BoxDomain
    return BoxDomain(np.zeros(3), np.ones(3))


@pytest.fixture(scope="module")
def space(box):
    return Lebesgue3D(box)


@pytest.fixture(scope="module")
def provider(box):
    from pygeoinf3d.providers.tensor_product import BoxLaplacianSpectrumProvider
    return BoxLaplacianSpectrumProvider(box, truncation=(4, 4, 4), bc_type="neumann")


@pytest.fixture(scope="module")
def sobolev_bessel(space, provider):
    from pygeoinf3d.operators.bessel_sobolev_box import BesselSobolevBox
    w = BesselSobolevBox(space, provider, shift=1.0, order=4.0)
    w_inv = BesselSobolevBox(space, provider, shift=1.0, order=-4.0)
    return Sobolev3D(space, w, w_inv)


@pytest.fixture(scope="module")
def path_straight():
    """A straight path across the unit box along x-axis."""
    from pygeoinf3d.core.paths import Path3D
    start = np.array([0.1, 0.5, 0.5])
    end = np.array([0.9, 0.5, 0.5])
    return Path3D.from_polyline(np.array([start, end]))


@pytest.fixture(scope="module")
def path_diagonal():
    from pygeoinf3d.core.paths import Path3D
    return Path3D.from_polyline(
        np.array([[0.1, 0.1, 0.1], [0.9, 0.9, 0.9]])
    )


@pytest.fixture(scope="module")
def form_straight(sobolev_bessel, path_straight):
    from pygeoinf3d.spaces.forms import PathIntegralFunctional
    return PathIntegralFunctional(sobolev_bessel, path_straight, n_quad=50)


# ---------------------------------------------------------------------------
# Reference: loop-based synthesis
# ---------------------------------------------------------------------------

def _manual_loop_representer(inv_w, coeffs, nus, pts):
    """Reference: loop-based synthesis.  pts must be (n, 3)."""
    total = np.zeros(len(pts))
    for ck, nuk, k in zip(coeffs, nus, range(len(coeffs))):
        phi_k = inv_w.get_eigenfunction(k)
        total = total + ck * nuk * np.asarray(phi_k(pts), dtype=float)
    return total


# ---------------------------------------------------------------------------
# TestFastPathAnalysisAndSynthesis
# ---------------------------------------------------------------------------

class TestFastPathAnalysisAndSynthesis:
    """PathIntegralFunctional + BoxLaplacianSpectrumProvider: vectorized both
    analysis (coefficient computation) and synthesis (evaluate closure)."""

    def test_from_dual_returns_function3d(self, sobolev_bessel, form_straight):
        """from_dual(PathIntegralFunctional) returns a Function3D."""
        r = sobolev_bessel.from_dual(form_straight)
        assert isinstance(r, Function3D)

    def test_fast_path_matches_loop_reference_batch(
        self, sobolev_bessel, form_straight
    ):
        """Vectorized from_dual agrees with manual spectral loop at rtol=1e-8."""
        inv_w = sobolev_bessel.inverse_weight
        n = inv_w.n_modes

        coeffs_ref = np.array([float(form_straight(inv_w.get_eigenfunction(k))) for k in range(n)])
        nus_ref = np.array([float(inv_w.get_eigenvalue(k)) for k in range(n)])

        pts = np.array([
            [0.25, 0.5, 0.5],
            [0.5, 0.5, 0.5],
            [0.75, 0.5, 0.5],
        ])
        ref = _manual_loop_representer(inv_w, coeffs_ref, nus_ref, pts)

        r = sobolev_bessel.from_dual(form_straight)
        got = r(pts)

        npt.assert_allclose(got, ref, rtol=1e-8)

    def test_fast_path_single_point_returns_scalar(
        self, sobolev_bessel, form_straight
    ):
        """Evaluating at a single (3,) point returns a Python float (scalar)."""
        r = sobolev_bessel.from_dual(form_straight)
        pt = np.array([0.5, 0.5, 0.5])
        val = r(pt)
        assert np.ndim(val) == 0, f"Expected scalar, got shape {np.shape(val)}"

    def test_fast_path_batch_points_returns_array(
        self, sobolev_bessel, form_straight
    ):
        """Evaluating at (n, 3) points returns an array of shape (n,)."""
        r = sobolev_bessel.from_dual(form_straight)
        pts = np.array([
            [0.1, 0.5, 0.5],
            [0.5, 0.5, 0.5],
            [0.9, 0.5, 0.5],
        ])
        val = r(pts)
        assert np.shape(val) == (3,), f"Expected shape (3,), got {np.shape(val)}"

    def test_fast_path_representer_identity(
        self, sobolev_bessel, form_straight
    ):
        """ell_gamma(r_gamma) approx ||r_gamma||_H^2 (representer identity)."""
        r = sobolev_bessel.from_dual(form_straight)
        lhs = float(form_straight(r))
        rhs = sobolev_bessel.inner_product(r, r)
        npt.assert_allclose(lhs, rhs, rtol=0.10)

    def test_fast_path_diagonal_path(
        self, sobolev_bessel, path_diagonal
    ):
        """Fast path works for a diagonal path (not axis-aligned)."""
        from pygeoinf3d.spaces.forms import PathIntegralFunctional
        form = PathIntegralFunctional(sobolev_bessel, path_diagonal, n_quad=50)
        r = sobolev_bessel.from_dual(form)
        assert isinstance(r, Function3D)
        pts = np.array([[0.5, 0.5, 0.5], [0.3, 0.3, 0.3]])
        vals = r(pts)
        assert np.all(np.isfinite(vals))


# ---------------------------------------------------------------------------
# TestFastSynthesisGenericForm
# ---------------------------------------------------------------------------

class TestFastSynthesisGenericForm:
    """Any LinearForm subclass (not PathIntegralFunctional) + BLSP provider ->
    vectorized synthesis (alpha @ Phi), loop analysis."""

    def test_generic_form_vectorized_synthesis_matches_loop(
        self, sobolev_bessel
    ):
        """A custom LinearForm subclass (not PathIntegralFunctional) with BLSP
        provider gives the same result as the manual loop reference."""
        inv_w = sobolev_bessel.inverse_weight
        n = inv_w.n_modes

        center_form = _PointEvalLinearForm(
            sobolev_bessel, np.array([0.5, 0.5, 0.5])
        )

        # Reference via loop
        coeffs_ref = np.array([
            float(center_form(inv_w.get_eigenfunction(k))) for k in range(n)
        ])
        nus_ref = np.array([float(inv_w.get_eigenvalue(k)) for k in range(n)])
        pts = np.array([[0.2, 0.4, 0.6], [0.7, 0.3, 0.5]])
        ref = _manual_loop_representer(inv_w, coeffs_ref, nus_ref, pts)

        r = sobolev_bessel.from_dual(center_form)
        got = r(pts)
        npt.assert_allclose(got, ref, rtol=1e-8)


# ---------------------------------------------------------------------------
# TestFallbackLoopPath
# ---------------------------------------------------------------------------

class TestFallbackLoopPath:
    """When inverse_weight has no BoxLaplacianSpectrumProvider, the original
    loop fallback is used.  Use PathIntegralFunctional (which has its own
    valid _mapping_impl) to exercise the analysis and synthesis loops."""

    def test_fallback_path_used_when_no_blsp(self, box):
        """A SpectralOperator3D NOT backed by BLSP falls back to the loop path."""
        from pygeoinf.linear_operators import LinearOperator as _LO
        from pygeoinf3d.spaces.forms import PathIntegralFunctional
        from pygeoinf3d.core.paths import Path3D

        space = Lebesgue3D(box)
        inv_w_no_blsp = _make_minimal_spectral_no_provider(space, box, n_modes=2)
        w_identity = _LO(space, space, lambda f: f, adjoint_mapping=lambda f: f)
        sob = Sobolev3D(space, w_identity, inv_w_no_blsp)

        path = Path3D.from_polyline(np.array([[0.1, 0.5, 0.5], [0.9, 0.5, 0.5]]))
        form = PathIntegralFunctional(sob, path, n_quad=20)

        r = sob.from_dual(form)
        assert isinstance(r, Function3D)

        pts = np.array([[0.2, 0.3, 0.4], [0.6, 0.7, 0.8]])
        vals = r(pts)
        assert np.all(np.isfinite(vals))
        assert vals.shape == (2,)

    def test_fallback_single_point_returns_scalar(self, box):
        """Fallback path single-point evaluation returns scalar."""
        from pygeoinf.linear_operators import LinearOperator as _LO
        from pygeoinf3d.spaces.forms import PathIntegralFunctional
        from pygeoinf3d.core.paths import Path3D

        space = Lebesgue3D(box)
        inv_w_no_blsp = _make_minimal_spectral_no_provider(space, box, n_modes=1)
        w_identity = _LO(space, space, lambda f: f, adjoint_mapping=lambda f: f)
        sob = Sobolev3D(space, w_identity, inv_w_no_blsp)

        path = Path3D.from_polyline(np.array([[0.1, 0.5, 0.5], [0.9, 0.5, 0.5]]))
        form = PathIntegralFunctional(sob, path, n_quad=20)

        r = sob.from_dual(form)

        pt = np.array([0.5, 0.5, 0.5])
        val = r(pt)
        assert np.ndim(val) == 0, f"Expected scalar, got {np.shape(val)}"


# ---------------------------------------------------------------------------
# TestBranch1Regression
# ---------------------------------------------------------------------------

class TestBranch1Regression:
    """Verify Branch 1 (VolumeKernel3D) is unaffected by Phase 4 changes."""

    def test_volume_kernel_round_trip(self, sobolev_bessel, box):
        """from_dual(to_dual(f)) approx f for a smooth function."""
        f = Function3D(
            box,
            evaluate_callable=lambda x: np.ones(len(x)) if x.ndim == 2 else 1.0,
        )
        fp = sobolev_bessel.to_dual(f)
        from pygeoinf3d.spaces.forms import VolumeKernel3D
        assert isinstance(fp, VolumeKernel3D)

        f_rec = sobolev_bessel.from_dual(fp)
        pts = np.array([
            [0.25, 0.25, 0.25],
            [0.5, 0.5, 0.5],
            [0.75, 0.75, 0.75],
        ])
        npt.assert_allclose(f_rec(pts), f(pts), atol=0.05)

    def test_volume_kernel_single_point(self, sobolev_bessel, box):
        """Branch 1 single-point evaluation should still work."""
        f = Function3D(
            box,
            evaluate_callable=lambda x: np.ones(len(x)) if x.ndim == 2 else 1.0,
        )
        fp = sobolev_bessel.to_dual(f)
        f_rec = sobolev_bessel.from_dual(fp)
        pt = np.array([0.5, 0.5, 0.5])
        val = f_rec(pt)
        npt.assert_allclose(float(val), 1.0, atol=0.1)
