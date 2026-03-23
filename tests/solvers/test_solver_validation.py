"""Tests for error message quality and runtime validation in solvers (Phase 3).

Verifies:
- SpectralGramSolver raises TypeError with actionable message for wrong space type
- SpectralGramSolver raises TypeError with actionable message for wrong operator type
- SpectralGramSolver raises ValueError for mixed n_quad values
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
)
from pygeoinf3d.spaces.forms import PathIntegralFunctional
from pygeoinf3d.solvers.spectral_gram import SpectralGramSolver


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
        unit_box, truncation=(3, 3, 3), bc_type="neumann"
    )


@pytest.fixture
def sobolev_space(unit_box, lebesgue_space, provider):
    W = BesselSobolevBox(lebesgue_space, provider, shift=1.0, order=1.75)
    W_inv = BesselSobolevBox(lebesgue_space, provider, shift=1.0, order=-1.75)
    return Sobolev3D(lebesgue_space, W, W_inv)


@pytest.fixture
def two_forms(unit_box, sobolev_space):
    """Two straight-line path forms with n_quad=20."""
    rng = np.random.default_rng(0)
    paths = [
        Path3D.from_polyline(
            np.array([rng.uniform(0.01, 0.99, 3), rng.uniform(0.01, 0.99, 3)]),
            domain=unit_box,
        )
        for _ in range(2)
    ]
    return [PathIntegralFunctional(sobolev_space, p, n_quad=20) for p in paths]


# ---------------------------------------------------------------------------
# Error message quality tests
# ---------------------------------------------------------------------------


def test_spectral_gram_wrong_space_type_error_message(unit_box, lebesgue_space, two_forms):
    """Passing Lebesgue3D instead of Sobolev3D should raise TypeError mentioning SOLAOperator3D."""
    with pytest.raises(TypeError) as exc_info:
        SpectralGramSolver(lebesgue_space, two_forms)
    assert "SOLAOperator3D" in str(exc_info.value), (
        f"Expected 'SOLAOperator3D' in error message, got: {exc_info.value}"
    )


def test_spectral_gram_wrong_operator_type_error_message(unit_box, lebesgue_space, two_forms):
    """Sobolev3D with a non-SpectralOperator3D weight should raise TypeError mentioning BesselSobolevBox."""
    from pygeoinf.linear_operators import LinearOperator

    # Create a plain LinearOperator (identity) that is NOT a SpectralOperator3D.
    identity = LinearOperator(lebesgue_space, lebesgue_space, lambda f: f)
    sobolev_bad = Sobolev3D(lebesgue_space, identity, identity)

    # Build forms on the bad Sobolev space
    rng = np.random.default_rng(1)
    paths = [
        Path3D.from_polyline(
            np.array([rng.uniform(0.01, 0.99, 3), rng.uniform(0.01, 0.99, 3)]),
            domain=unit_box,
        )
        for _ in range(2)
    ]
    forms_bad = [PathIntegralFunctional(sobolev_bad, p, n_quad=20) for p in paths]

    with pytest.raises(TypeError) as exc_info:
        SpectralGramSolver(sobolev_bad, forms_bad)
    assert "BesselSobolevBox" in str(exc_info.value), (
        f"Expected 'BesselSobolevBox' in error message, got: {exc_info.value}"
    )


def test_spectral_gram_mixed_nquad_raises(unit_box, sobolev_space):
    """Forms with different n_quad values should raise ValueError mentioning 'n_quad'."""
    rng = np.random.default_rng(2)

    def make_path():
        return Path3D.from_polyline(
            np.array([rng.uniform(0.01, 0.99, 3), rng.uniform(0.01, 0.99, 3)]),
            domain=unit_box,
        )

    form_a = PathIntegralFunctional(sobolev_space, make_path(), n_quad=20)
    form_b = PathIntegralFunctional(sobolev_space, make_path(), n_quad=40)

    with pytest.raises(ValueError) as exc_info:
        SpectralGramSolver(sobolev_space, [form_a, form_b])
    assert "n_quad" in str(exc_info.value), (
        f"Expected 'n_quad' in error message, got: {exc_info.value}"
    )
