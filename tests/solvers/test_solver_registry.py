"""Tests for the solver registry (Phase 1 of solver-compatibility-framework)."""

import pytest

from pygeoinf3d import (
    Sobolev3D,
    SOLVER_REGISTRY,
    SolverSpec,
    SpectralGramSolver,
)
from pygeoinf3d.operators.base import SpectralOperator3D
from pygeoinf3d.spaces.forms import PathIntegralFunctional


# ---------------------------------------------------------------------------
# Registry population
# ---------------------------------------------------------------------------

def test_registry_has_four_entries():
    assert len(SOLVER_REGISTRY.list_all()) == 4


# ---------------------------------------------------------------------------
# SpectralGramSolver spec
# ---------------------------------------------------------------------------

def test_spectral_gram_spec_name():
    spec = SOLVER_REGISTRY.find_by_name("SpectralGramSolver")
    assert isinstance(spec, SolverSpec)


def test_spectral_gram_spec_solver_class():
    spec = SOLVER_REGISTRY.find_by_name("SpectralGramSolver")
    assert spec.solver_class is SpectralGramSolver


def test_spectral_gram_spec_space_types():
    spec = SOLVER_REGISTRY.find_by_name("SpectralGramSolver")
    assert Sobolev3D in spec.space_types


def test_spectral_gram_spec_operator_types():
    spec = SOLVER_REGISTRY.find_by_name("SpectralGramSolver")
    assert SpectralOperator3D in spec.operator_types


def test_spectral_gram_spec_functional_types():
    spec = SOLVER_REGISTRY.find_by_name("SpectralGramSolver")
    assert PathIntegralFunctional in spec.functional_types


def test_spectral_gram_spec_purpose():
    spec = SOLVER_REGISTRY.find_by_name("SpectralGramSolver")
    assert spec.purpose == "inversion"


def test_spectral_gram_spec_performance_tier():
    spec = SOLVER_REGISTRY.find_by_name("SpectralGramSolver")
    assert spec.performance_tier == "fast"


# ---------------------------------------------------------------------------
# SOLAOperator3D spec
# ---------------------------------------------------------------------------

def test_sola_spec_name():
    spec = SOLVER_REGISTRY.find_by_name("SOLAOperator3D")
    assert isinstance(spec, SolverSpec)


def test_sola_spec_purpose():
    spec = SOLVER_REGISTRY.find_by_name("SOLAOperator3D")
    assert spec.purpose == "measurement_operator"


# ---------------------------------------------------------------------------
# KLSampler3D spec
# ---------------------------------------------------------------------------

def test_kl_spec_name():
    spec = SOLVER_REGISTRY.find_by_name("KLSampler3D")
    assert isinstance(spec, SolverSpec)


def test_kl_spec_purpose():
    spec = SOLVER_REGISTRY.find_by_name("KLSampler3D")
    assert spec.purpose == "sampling"


# ---------------------------------------------------------------------------
# find_by_purpose
# ---------------------------------------------------------------------------

def test_find_by_purpose_inversion():
    results = SOLVER_REGISTRY.find_by_purpose("inversion")
    solver_classes = [s.solver_class for s in results]
    assert SpectralGramSolver in solver_classes


def test_find_by_purpose_sampling():
    from pygeoinf3d.sampling.kl_sampler import KLSampler3D
    results = SOLVER_REGISTRY.find_by_purpose("sampling")
    solver_classes = [s.solver_class for s in results]
    assert KLSampler3D in solver_classes


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

def test_find_by_name_not_found():
    with pytest.raises(KeyError):
        SOLVER_REGISTRY.find_by_name("NonExistentSolver")
