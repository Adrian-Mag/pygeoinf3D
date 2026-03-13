"""Tests for the pygeoinf3d.providers.function_providers package scaffold.

Phase 1 scope — import/export locks and provider-protocol contract:

- The ``function_providers`` sub-package is importable.
- It exposes ``__all__``.
- ``FunctionProvider3D`` is re-exported from it (subclass convenience).
- ``pygeoinf3d.providers.function_providers`` is accessible via the parent
  ``providers`` package.
- A minimal concrete subclass satisfies the ``FunctionProvider3D`` contract
  (``dim`` and ``get_function`` are callable with correct return type).
"""

from __future__ import annotations

import importlib
import types

import numpy as np
import pytest

from pygeoinf3d.core.functions import Function3D
from pygeoinf3d.providers.base import FunctionProvider3D
from pygeoinf3d.geometries.box.domain import BoxDomain


# ---------------------------------------------------------------------------
# Helpers — minimal concrete subclass used as a test double
# ---------------------------------------------------------------------------

def _constant_function(domain: BoxDomain, value: float) -> Function3D:
    """Return a Function3D that evaluates to *value* everywhere."""
    def _eval(pts):
        pts = np.asarray(pts)
        if pts.ndim == 1:
            return float(value)
        return np.full(pts.shape[0], float(value))

    return Function3D(domain, evaluate_callable=_eval)


class _StubProvider(FunctionProvider3D):
    """Minimal concrete provider for contract testing."""

    def __init__(self, domain: BoxDomain, n: int):
        self._domain = domain
        self._n = n

    @property
    def dim(self) -> int:
        return self._n

    def get_function(self, i: int) -> Function3D:
        if not (0 <= i < self._n):
            raise IndexError(f"Index {i} out of range [0, {self._n})")
        return _constant_function(self._domain, float(i))


# ---------------------------------------------------------------------------
# Package-level import / export tests
# ---------------------------------------------------------------------------

class TestFunctionProvidersPackageImport:
    """Lock that the sub-package is importable and well-formed."""

    def test_package_importable(self):
        """pygeoinf3d.providers.function_providers must be importable."""
        mod = importlib.import_module(
            "pygeoinf3d.providers.function_providers"
        )
        assert isinstance(mod, types.ModuleType)

    def test_package_has_all(self):
        """The sub-package must define __all__."""
        mod = importlib.import_module(
            "pygeoinf3d.providers.function_providers"
        )
        assert hasattr(mod, "__all__"), (
            "function_providers must define __all__"
        )
        assert isinstance(mod.__all__, list)
        assert mod.__all__ == ["FunctionProvider3D"]

    def test_function_provider_3d_re_exported(self):
        """FunctionProvider3D must be accessible from the sub-package."""
        from pygeoinf3d.providers.function_providers import (
            FunctionProvider3D as FP,
        )

        assert FP is FunctionProvider3D

    def test_accessible_via_parent_providers(self):
        """pygeoinf3d.providers.function_providers accessible as attribute."""
        import pygeoinf3d.providers.function_providers  # noqa: F401
        import pygeoinf3d.providers as providers_pkg

        assert hasattr(providers_pkg, "function_providers")

    def test_all_symbols_importable(self):
        """Every name in __all__ must importable from the sub-package."""
        mod = importlib.import_module(
            "pygeoinf3d.providers.function_providers"
        )
        for name in mod.__all__:
            assert hasattr(mod, name), (
                f"'{name}' is listed in __all__ but not defined in the module"
            )


# ---------------------------------------------------------------------------
# FunctionProvider3D protocol contract tests (using the test double)
# ---------------------------------------------------------------------------

@pytest.fixture
def unit_box():
    return BoxDomain(np.zeros(3), np.ones(3))


@pytest.fixture
def stub_provider(unit_box):
    return _StubProvider(unit_box, n=4)


class TestFunctionProvider3DContract:
    """Generic contract that every FunctionProvider3D subclass must satisfy."""

    def test_dim_is_positive_int(self, stub_provider):
        assert isinstance(stub_provider.dim, int)
        assert stub_provider.dim > 0

    def test_get_function_returns_function3d(self, stub_provider):
        fn = stub_provider.get_function(0)
        assert isinstance(fn, Function3D)

    def test_get_function_all_indices(self, stub_provider):
        for i in range(stub_provider.dim):
            fn = stub_provider.get_function(i)
            assert isinstance(fn, Function3D)

    def test_get_function_out_of_range_raises(self, stub_provider):
        with pytest.raises(IndexError):
            stub_provider.get_function(stub_provider.dim)

    def test_get_function_negative_raises(self, stub_provider):
        with pytest.raises(IndexError):
            stub_provider.get_function(-1)

    def test_returned_function_is_callable(self, stub_provider):
        fn = stub_provider.get_function(0)
        pts = np.array([[0.5, 0.5, 0.5]])
        result = fn(pts)
        assert result is not None

    def test_returned_function_has_domain(self, stub_provider, unit_box):
        fn = stub_provider.get_function(0)
        assert fn.domain is unit_box
