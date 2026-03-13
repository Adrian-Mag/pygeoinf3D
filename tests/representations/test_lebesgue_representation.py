"""Tests for pygeoinf3d.representations.lebesgue.LebesgueRepresentation3D.

Phase 1 — Representation Boundary and API Contracts.

Covers:
- LebesgueRepresentation3D construction and attribute access
- dim delegated from provider
- coefficient_space is EuclideanSpace(dim)
- to_components/from_components delegate to backend
- Domain-compatibility validation
- Rejection of non-Lebesgue3D spaces
- CoefficientRepresentation3D ABC cannot be instantiated directly
"""

from __future__ import annotations

import numpy as np
import pytest

from pygeoinf import EuclideanSpace, Ball

from pygeoinf3d.core.config import AxisAlignedBoundingBox
from pygeoinf3d.core.region import Region3D
from pygeoinf3d.core.domain_base import Domain3D
from pygeoinf3d.core.boundary import Boundary3D
from pygeoinf3d.core.functions import Function3D


# ---------------------------------------------------------------------------
# Minimal mock domain (same pattern as test_lebesgue.py)
# ---------------------------------------------------------------------------

class _MockBoundary(Boundary3D):
    def contains(self, point: np.ndarray) -> bool:
        return False


class _MockDomain(Domain3D):
    _SEED = 99
    _N = 10_000

    def sample_interior(self, n, *, rng=None):
        rng = np.random.default_rng(rng)
        return rng.uniform(0.0, 1.0, size=(n, 3))

    def integrate_volume(self, f, *, method=None, **kwargs):
        n = kwargs.get("n", self._N)
        seed = kwargs.get("seed", self._SEED)
        pts = self.sample_interior(n, rng=seed)
        return float(np.mean(f(pts)))

    @property
    def boundary(self):
        return _MockBoundary(self)


def _make_region():
    space = EuclideanSpace(3)
    ball = Ball(space, np.array([0.5, 0.5, 0.5]), 0.5)
    bbox = AxisAlignedBoundingBox(low=np.zeros(3), high=np.ones(3))
    return Region3D(ball, bounding_box=bbox)


def _make_domain():
    return _MockDomain(_make_region())


# ---------------------------------------------------------------------------
# Dummy provider and backend used as minimal stand-ins for Phase 1
# ---------------------------------------------------------------------------

class _DummyProvider:
    """Minimal provider duck-type: exposes domain and dim."""

    def __init__(self, domain, *, dim: int):
        self.domain = domain
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim


class _DummyBackend:
    """Minimal backend stub: returns trivial arrays/functions."""

    def to_components(self, space, provider, f) -> np.ndarray:
        return np.ones(provider.dim, dtype=float)

    def from_components(self, space, provider, c) -> Function3D:
        return Function3D(
            space.function_domain,
            evaluate_callable=lambda x: np.zeros(len(x)) if np.asarray(x).ndim == 2 else 0.0,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_space_and_stub_rep(n: int = 6):
    from pygeoinf3d.spaces.lebesgue import Lebesgue3D
    from pygeoinf3d.representations.lebesgue import LebesgueRepresentation3D

    dom = _make_domain()
    space = Lebesgue3D(dom)
    provider = _DummyProvider(dom, dim=n)
    backend = _DummyBackend()
    rep = LebesgueRepresentation3D(space, provider, backend)
    return space, dom, provider, backend, rep


# ===========================================================================
# CoefficientRepresentation3D ABC
# ===========================================================================

class TestCoefficientRepresentation3DABC:
    def test_cannot_instantiate_abstract_base_directly(self):
        from pygeoinf3d.representations.base import CoefficientRepresentation3D
        with pytest.raises(TypeError):
            CoefficientRepresentation3D()  # type: ignore[abstract]

    def test_concrete_subclass_must_implement_space(self):
        from pygeoinf3d.representations.base import CoefficientRepresentation3D

        class _Incomplete(CoefficientRepresentation3D):
            # Missing space, dim, to_components, from_components
            pass

        with pytest.raises(TypeError):
            _Incomplete()  # type: ignore[abstract]

    def test_concrete_subclass_satisfies_abc(self):
        """A fully-implemented subclass should be instantiable."""
        from pygeoinf3d.representations.base import CoefficientRepresentation3D
        from pygeoinf3d.spaces.lebesgue import Lebesgue3D

        dom = _make_domain()
        space = Lebesgue3D(dom)

        class _MinimalRep(CoefficientRepresentation3D):
            @property
            def space(self):
                return space

            @property
            def dim(self):
                return 3

            def to_components(self, f):
                return np.zeros(3)

            def from_components(self, c):
                return Function3D(
                    space.function_domain,
                    evaluate_callable=lambda x: np.zeros(len(x)) if np.asarray(x).ndim == 2 else 0.0,
                )

        rep = _MinimalRep()
        assert rep.dim == 3


# ===========================================================================
# LebesgueRepresentation3D — Construction and attributes
# ===========================================================================

class TestLebesgueRepresentation3DConstruction:
    def test_construction_succeeds(self):
        from pygeoinf3d.representations.lebesgue import LebesgueRepresentation3D
        dom = _make_domain()
        from pygeoinf3d.spaces.lebesgue import Lebesgue3D
        space = Lebesgue3D(dom)
        provider = _DummyProvider(dom, dim=8)
        backend = _DummyBackend()
        rep = LebesgueRepresentation3D(space, provider, backend)
        assert rep is not None

    def test_space_attribute(self):
        _, dom, _, _, rep = _make_space_and_stub_rep()
        from pygeoinf3d.spaces.lebesgue import Lebesgue3D
        space = Lebesgue3D(dom)
        from pygeoinf3d.representations.lebesgue import LebesgueRepresentation3D
        provider = _DummyProvider(dom, dim=6)
        backend = _DummyBackend()
        rep2 = LebesgueRepresentation3D(space, provider, backend)
        assert rep2.space is space

    def test_provider_attribute(self):
        from pygeoinf3d.spaces.lebesgue import Lebesgue3D
        from pygeoinf3d.representations.lebesgue import LebesgueRepresentation3D
        dom = _make_domain()
        space = Lebesgue3D(dom)
        provider = _DummyProvider(dom, dim=3)
        backend = _DummyBackend()
        rep = LebesgueRepresentation3D(space, provider, backend)
        assert rep.provider is provider

    def test_backend_attribute(self):
        from pygeoinf3d.spaces.lebesgue import Lebesgue3D
        from pygeoinf3d.representations.lebesgue import LebesgueRepresentation3D
        dom = _make_domain()
        space = Lebesgue3D(dom)
        provider = _DummyProvider(dom, dim=3)
        backend = _DummyBackend()
        rep = LebesgueRepresentation3D(space, provider, backend)
        assert rep.backend is backend

    def test_rejects_non_lebesgue_space(self):
        from pygeoinf3d.representations.lebesgue import LebesgueRepresentation3D
        from pygeoinf import EuclideanSpace as ES
        bad_space = ES(3)
        dom = _make_domain()
        provider = _DummyProvider(dom, dim=2)
        backend = _DummyBackend()
        with pytest.raises(TypeError):
            LebesgueRepresentation3D(bad_space, provider, backend)

    def test_rejects_mismatched_provider_domain(self):
        from pygeoinf3d.representations.lebesgue import LebesgueRepresentation3D
        from pygeoinf3d.spaces.lebesgue import Lebesgue3D
        dom1 = _make_domain()
        dom2 = _make_domain()
        space = Lebesgue3D(dom1)
        provider = _DummyProvider(dom2, dim=4)  # different domain instance
        backend = _DummyBackend()
        with pytest.raises(ValueError):
            LebesgueRepresentation3D(space, provider, backend)

    def test_accepts_provider_without_domain_attribute(self):
        """Provider without a .domain attribute is accepted (permissive validation)."""
        from pygeoinf3d.representations.lebesgue import LebesgueRepresentation3D
        from pygeoinf3d.spaces.lebesgue import Lebesgue3D

        class _NoDomainProvider:
            dim = 5

        dom = _make_domain()
        space = Lebesgue3D(dom)
        provider = _NoDomainProvider()
        backend = _DummyBackend()
        rep = LebesgueRepresentation3D(space, provider, backend)
        assert rep.dim == 5


# ===========================================================================
# LebesgueRepresentation3D — dim and coefficient_space
# ===========================================================================

class TestLebesgueRepresentation3DDim:
    def test_dim_delegates_to_provider(self):
        _, _, _, _, rep = _make_space_and_stub_rep(n=7)
        assert rep.dim == 7

    def test_dim_changes_with_provider_dim(self):
        from pygeoinf3d.spaces.lebesgue import Lebesgue3D
        from pygeoinf3d.representations.lebesgue import LebesgueRepresentation3D
        dom = _make_domain()
        space = Lebesgue3D(dom)
        for n in [1, 4, 12, 100]:
            provider = _DummyProvider(dom, dim=n)
            rep = LebesgueRepresentation3D(space, provider, _DummyBackend())
            assert rep.dim == n

    def test_coefficient_space_is_euclidean(self):
        from pygeoinf import EuclideanSpace
        _, _, _, _, rep = _make_space_and_stub_rep(n=5)
        cs = rep.coefficient_space
        assert isinstance(cs, EuclideanSpace)
        assert cs.dim == 5

    def test_coefficient_space_dim_matches_representation_dim(self):
        _, _, _, _, rep = _make_space_and_stub_rep(n=9)
        assert rep.coefficient_space.dim == rep.dim


# ===========================================================================
# LebesgueRepresentation3D — to_components / from_components delegation
# ===========================================================================

class TestLebesgueRepresentation3DDelegation:
    def test_to_components_returns_array(self):
        space, dom, _, _, rep = _make_space_and_stub_rep(n=4)
        f = Function3D(dom, evaluate_callable=lambda x: np.ones(len(x)) if np.asarray(x).ndim == 2 else 1.0)
        c = rep.to_components(f)
        assert isinstance(c, np.ndarray)

    def test_to_components_length_matches_dim(self):
        space, dom, _, _, rep = _make_space_and_stub_rep(n=6)
        f = Function3D(dom, evaluate_callable=lambda x: np.ones(len(x)) if np.asarray(x).ndim == 2 else 1.0)
        c = rep.to_components(f)
        assert len(c) == 6

    def test_to_components_calls_backend(self):
        """Backend stub returns np.ones(dim), so check shape and values."""
        space, dom, provider, _, rep = _make_space_and_stub_rep(n=3)
        f = Function3D(dom, evaluate_callable=lambda x: np.zeros(len(x)) if np.asarray(x).ndim == 2 else 0.0)
        c = rep.to_components(f)
        np.testing.assert_array_equal(c, np.ones(3))

    def test_from_components_returns_function3d(self):
        space, dom, _, _, rep = _make_space_and_stub_rep(n=4)
        c = np.zeros(4)
        f_syn = rep.from_components(c)
        assert isinstance(f_syn, Function3D)

    def test_from_components_function_on_correct_domain(self):
        space, dom, _, _, rep = _make_space_and_stub_rep(n=4)
        c = np.zeros(4)
        f_syn = rep.from_components(c)
        assert f_syn.domain is dom

    def test_from_components_backend_stub_returns_zero(self):
        space, dom, _, _, rep = _make_space_and_stub_rep(n=4)
        c = np.array([1.0, 2.0, 3.0, 4.0])
        f_syn = rep.from_components(c)
        pt = np.array([[0.5, 0.5, 0.5]])
        val = f_syn(pt)
        np.testing.assert_allclose(float(val[0]) if np.asarray(val).ndim > 0 else float(val), 0.0, atol=1e-15)

    def test_round_trip_type_contract(self):
        """to_components → from_components → type check only (no numerics yet)."""
        space, dom, _, _, rep = _make_space_and_stub_rep(n=5)
        f = Function3D(dom, evaluate_callable=lambda x: np.ones(len(x)) if np.asarray(x).ndim == 2 else 1.0)
        c = rep.to_components(f)
        f_back = rep.from_components(c)
        assert isinstance(f_back, Function3D)


# ===========================================================================
# CoefficientRepresentation3D — coefficient_space default implementation
# ===========================================================================

class TestCoefficientRepresentation3DCoeffSpace:
    def test_coefficient_space_from_base_class(self):
        """The base class coefficient_space property should use EuclideanSpace(dim)."""
        from pygeoinf3d.representations.base import CoefficientRepresentation3D
        from pygeoinf3d.spaces.lebesgue import Lebesgue3D
        from pygeoinf import EuclideanSpace

        dom = _make_domain()
        space = Lebesgue3D(dom)

        class _ConcreteRep(CoefficientRepresentation3D):
            @property
            def space(self):
                return space

            @property
            def dim(self):
                return 7

            def to_components(self, f):
                return np.zeros(7)

            def from_components(self, c):
                return Function3D(
                    space.function_domain,
                    evaluate_callable=lambda x: np.zeros(len(x)) if np.asarray(x).ndim == 2 else 0.0,
                )

        rep = _ConcreteRep()
        cs = rep.coefficient_space
        assert isinstance(cs, EuclideanSpace)
        assert cs.dim == 7
