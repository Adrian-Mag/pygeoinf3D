"""Tests for pygeoinf3d.solvers.router.SolverRouter (Phase 2).

Verifies:
- recommend() returns a SolverRecommendation object
- SpectralGramSolver recommended for Sobolev3D + PathIntegralFunctional
- SOLAOperator3D recommended for Lebesgue3D + VolumeKernel3D
- SOLAOperator3D recommended for Sobolev3D + VolumeKernel3D (with warning)
- Warning present when Sobolev3D used with non-PathIntegralFunctional forms
- recommend_sampler() recommends KLSampler3D for SpectralOperator3D
- recommend_sampler() returns no recommendation for non-SpectralOperator3D
- No recommendation for unknown form types
- list_available() returns 3 SolverSpec entries
"""

from __future__ import annotations

import numpy as np
import pytest

from pygeoinf3d import (
    BoxDomain,
    Lebesgue3D,
    Sobolev3D,
    BesselSobolevBox,
    BoxLaplacianSpectrumProvider,
    Path3D,
    Function3D,
)
from pygeoinf3d.spaces.forms import PathIntegralFunctional, VolumeKernel3D
from pygeoinf3d.solvers.router import SolverRouter, SolverRecommendation
from pygeoinf3d.solvers.registry import SolverSpec
from pygeoinf3d.solvers.spectral_gram import SpectralGramSolver
from pygeoinf3d.operators.sola import SOLAOperator3D
from pygeoinf3d.sampling.kl_sampler import KLSampler3D


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
def provider(unit_box):
    return BoxLaplacianSpectrumProvider(
        unit_box, truncation=(4, 4, 4), bc_type="neumann"
    )


@pytest.fixture
def sobolev_space(unit_box, lebesgue_space, provider):
    W = BesselSobolevBox(lebesgue_space, provider, shift=1.0, order=1.75)
    W_inv = BesselSobolevBox(lebesgue_space, provider, shift=1.0, order=-1.75)
    return Sobolev3D(lebesgue_space, W, W_inv)


@pytest.fixture
def path_forms(unit_box, sobolev_space):
    """3 random straight-line paths → PathIntegralFunctional forms."""
    rng = np.random.default_rng(7)
    sources = rng.uniform(0.05, 0.95, size=(3, 3))
    receivers = rng.uniform(0.05, 0.95, size=(3, 3))
    paths = [
        Path3D.from_polyline(np.array([sources[i], receivers[i]]), domain=unit_box)
        for i in range(3)
    ]
    return [PathIntegralFunctional(sobolev_space, p, n_quad=50) for p in paths]


@pytest.fixture
def volume_kernels_lebesgue(unit_box, lebesgue_space):
    """2 simple VolumeKernel3D forms on Lebesgue3D."""
    centers = np.array([[0.25, 0.25, 0.25], [0.75, 0.75, 0.75]])

    def make_gaussian(c):
        def g(x):
            x = np.asarray(x)
            return np.exp(-np.sum((x - c) ** 2, axis=-1) / 0.1)
        return Function3D(unit_box, evaluate_callable=g)

    return [
        VolumeKernel3D(lebesgue_space, make_gaussian(centers[i]))
        for i in range(2)
    ]


@pytest.fixture
def volume_kernels_sobolev(unit_box, sobolev_space):
    """2 simple VolumeKernel3D forms on Sobolev3D."""
    centers = np.array([[0.25, 0.25, 0.25], [0.75, 0.75, 0.75]])

    def make_gaussian(c):
        def g(x):
            x = np.asarray(x)
            return np.exp(-np.sum((x - c) ** 2, axis=-1) / 0.1)
        return Function3D(unit_box, evaluate_callable=g)

    return [
        VolumeKernel3D(sobolev_space, make_gaussian(centers[i]))
        for i in range(2)
    ]


@pytest.fixture
def router():
    return SolverRouter()


# ---------------------------------------------------------------------------
# Basic structure
# ---------------------------------------------------------------------------


def test_recommend_returns_recommendation_object(router, sobolev_space, path_forms):
    result = router.recommend(sobolev_space, path_forms)
    assert isinstance(result, SolverRecommendation)


def test_recommendation_has_reason_string(router, sobolev_space, path_forms):
    result = router.recommend(sobolev_space, path_forms)
    assert isinstance(result.reason, str)
    assert len(result.reason) > 0


# ---------------------------------------------------------------------------
# SpectralGramSolver fast path
# ---------------------------------------------------------------------------


def test_spectral_gram_recommended_for_sobolev_path_forms(
    router, sobolev_space, path_forms
):
    result = router.recommend(sobolev_space, path_forms)
    assert result.solver_spec is not None
    assert result.solver_spec.solver_class is SpectralGramSolver


def test_spectral_gram_spec_is_correct(router, sobolev_space, path_forms):
    result = router.recommend(sobolev_space, path_forms)
    assert result.solver_spec.name == "SpectralGramSolver"


# ---------------------------------------------------------------------------
# SOLAOperator3D general path
# ---------------------------------------------------------------------------


def test_sola_recommended_for_lebesgue_volume_kernels(
    router, lebesgue_space, volume_kernels_lebesgue
):
    result = router.recommend(lebesgue_space, volume_kernels_lebesgue)
    assert result.solver_spec is not None
    assert result.solver_spec.solver_class is SOLAOperator3D


def test_sola_recommended_for_sobolev_volume_kernels(
    router, sobolev_space, volume_kernels_sobolev
):
    result = router.recommend(sobolev_space, volume_kernels_sobolev)
    assert result.solver_spec is not None
    assert result.solver_spec.solver_class is SOLAOperator3D


def test_sola_warning_when_spectral_but_volume_kernels(
    router, sobolev_space, volume_kernels_sobolev
):
    """Warning is present when Sobolev3D used with VolumeKernel3D (not PathIntegral)."""
    result = router.recommend(sobolev_space, volume_kernels_sobolev)
    assert len(result.warnings) > 0
    # Should mention the faster alternative
    combined = " ".join(result.warnings)
    assert "SpectralGramSolver" in combined or "faster" in combined.lower()


def test_sola_recommendation_has_solver_spec(
    router, lebesgue_space, volume_kernels_lebesgue
):
    result = router.recommend(lebesgue_space, volume_kernels_lebesgue)
    assert result.solver_spec is not None
    assert isinstance(result.solver_spec, SolverSpec)


# ---------------------------------------------------------------------------
# No recommendation
# ---------------------------------------------------------------------------


def test_no_recommendation_for_unknown_forms(router, lebesgue_space):
    """Completely unknown form type → solver_spec is None."""

    class _FakeForm:
        pass

    result = router.recommend(lebesgue_space, [_FakeForm()])
    assert result.solver_spec is None
    assert "No compatible solver" in result.reason


# ---------------------------------------------------------------------------
# recommend_sampler
# ---------------------------------------------------------------------------


def test_recommend_sampler_for_spectral_operator(router, lebesgue_space, provider):
    from pygeoinf3d.operators.bessel_sobolev_box import BesselSobolevBox

    inv_lap = BesselSobolevBox(lebesgue_space, provider, shift=1.0, order=-1.0)
    result = router.recommend_sampler(inv_lap)
    assert result.solver_spec is not None
    assert result.solver_spec.solver_class is KLSampler3D


def test_recommend_sampler_for_non_spectral_returns_none(router):
    """A plain object (not SpectralOperator3D) → solver_spec is None."""

    class _PlainOp:
        pass

    result = router.recommend_sampler(_PlainOp())
    assert result.solver_spec is None


# ---------------------------------------------------------------------------
# list_available
# ---------------------------------------------------------------------------


def test_list_available_returns_four(router):
    specs = router.list_available()
    assert len(specs) == 4
