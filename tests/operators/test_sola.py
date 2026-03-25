"""Tests for pygeoinf3d.operators.sola.SOLAOperator3D (Phase 1 + 2 + 3 + 4).

Covers (Phase 1):
- Forward evaluation matches analytic integrals for known kernels
- Custom codomain realization via from_components is used
- Rejection of mismatched measurement counts
- Rejection of non-VolumeKernel3D kernels
- Rejection of kernels from a different Lebesgue3D instance
- Rejection of non-Lebesgue3D domain
- SOLAOperator3D is a LinearOperator instance

Covers (Phase 2):
- Dual mapping extracts codomain-dual coordinates and returns
    weighted sum of forms
- Adjoint identity holds for EuclideanSpace codomain (pairing test)
- Adjoint identity holds for MassWeightedHilbertSpace codomain

Covers (Phase 3):
- SOLAOperator3D accepts a KernelProvider3D (provider-backed construction)
- Provider-backed forward map matches sequence-backed forward map
- KernelProvider3D cache returns stable (identity-equal) kernel objects

Covers (Phase 4):
- Top-level pygeoinf3d exports SOLAOperator3D, FunctionProvider3D,
    KernelProvider3D, FunctionToKernelAdapter (names and identity)
- pygeoinf3d.__all__ contains all four SOLA symbols
- pygeoinf3d.operators exports SOLAOperator3D

Phase 4 Revision (Phase 4R):
- Mixed VolumeKernel3D + PathIntegralFunctional stacks are accepted
- _dual_mapping returns SumLinearForm for mixed/multi-path stacks
- Adjoint error test uses 2-form path stack (dual returns SumLinearForm)
"""

from __future__ import annotations

import pytest
import numpy as np
import numpy.testing as npt

from pygeoinf import EuclideanSpace, MassWeightedHilbertSpace
from pygeoinf.linear_forms import LinearForm
from pygeoinf.linear_operators import LinearOperator
from pygeoinf3d.geometries.box.domain import BoxDomain
from pygeoinf3d.spaces.lebesgue import Lebesgue3D
from pygeoinf3d.spaces.sobolev import Sobolev3D
from pygeoinf3d.spaces.forms import VolumeKernel3D
from pygeoinf3d.core.functions import Function3D
from pygeoinf3d.operators.sola import SOLAOperator3D


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def unit_box():
    return BoxDomain(np.zeros(3), np.ones(3))


@pytest.fixture
def space(unit_box):
    return Lebesgue3D(unit_box)


# ---------------------------------------------------------------------------
# Helper functions for building known Function3D objects
# ---------------------------------------------------------------------------

def const_fn(domain, c: float) -> Function3D:
    """Return a Function3D equal to the scalar constant c everywhere."""
    c_f = float(c)
    return Function3D(
        domain,
        evaluate_callable=lambda x: (
            np.full(len(x), c_f) if np.asarray(x).ndim == 2
            else c_f
        ),
    )


def linear_fn(domain, axis: int) -> Function3D:
    """Return a Function3D equal to x[axis] (linear in one coordinate)."""
    ax = int(axis)
    return Function3D(
        domain,
        evaluate_callable=lambda x: (
            np.asarray(x)[:, ax].copy() if np.asarray(x).ndim == 2
            else float(np.asarray(x)[ax])
        ),
    )


def poly_fn(domain, axis: int, power: int) -> Function3D:
    """Return a Function3D equal to x[axis]^power."""
    ax, pw = int(axis), int(power)
    return Function3D(
        domain,
        evaluate_callable=lambda x: (
            np.asarray(x)[:, ax] ** pw if np.asarray(x).ndim == 2
            else float(np.asarray(x)[ax]) ** pw
        ),
    )


# ---------------------------------------------------------------------------
# Test: forward evaluation matches analytic integrals on [0,1]^3
# ---------------------------------------------------------------------------

class TestSOLAForwardAnalytic:
    """Forward map produces values matching closed-form integrals."""

    def test_sola_forward_volume_kernels_matches_analytic_integrals(
        self, unit_box, space
    ):
        """
        Two kernels on f(x) = x₀ over [0,1]³:

        ∫_{[0,1]³} 1 · x₀ dV  = 0.5
        ∫_{[0,1]³} 2x₀ · x₀ dV = 2 ∫ x₀² dV = 2/3

        The tolerance is chosen to match the default midpoint quadrature
        (resolution=20), which is accurate to O(h²) = O(1/400) ≈ 2.5e-3.
        """
        k1 = const_fn(unit_box, 1.0)
        k2 = Function3D(
            unit_box,
            evaluate_callable=lambda x: (
                2.0 * np.asarray(x)[:, 0] if np.asarray(x).ndim == 2
                else 2.0 * float(np.asarray(x)[0])
            ),
        )
        m1 = VolumeKernel3D(space, k1)
        m2 = VolumeKernel3D(space, k2)

        codomain = EuclideanSpace(2)
        op = SOLAOperator3D(space, codomain, [m1, m2])

        f = linear_fn(unit_box, 0)   # f(x) = x₀
        y = op(f)

        npt.assert_allclose(y[0], 0.5,        rtol=5e-3, atol=5e-3)
        npt.assert_allclose(y[1], 2.0 / 3.0,  rtol=5e-3, atol=5e-3)

    def test_sola_forward_constant_kernel_constant_function(
        self, unit_box, space
    ):
        """
        ∫_{[0,1]³} c · 1 dV = c  (volume of unit box is 1).
        """
        for c in [0.0, 1.0, -2.5, np.pi]:
            k = const_fn(unit_box, 1.0)
            m = VolumeKernel3D(space, k)
            codomain = EuclideanSpace(1)
            op = SOLAOperator3D(space, codomain, [m])
            y = op(const_fn(unit_box, c))
            npt.assert_allclose(
                y[0], float(c), rtol=1e-10, atol=1e-10,
                err_msg=f"constant function c={c!r} failed",
            )

    def test_sola_forward_single_linear_measurement(self, unit_box, space):
        """
        Kernel k(x) = x₁, function f(x) = x₁:
        ∫ x₁² dV = 1/3  (over unit cube).
        """
        k = linear_fn(unit_box, 1)
        m = VolumeKernel3D(space, k)
        codomain = EuclideanSpace(1)
        op = SOLAOperator3D(space, codomain, [m])

        f = linear_fn(unit_box, 1)
        y = op(f)
        npt.assert_allclose(y[0], 1.0 / 3.0, rtol=5e-3, atol=5e-3)

    def test_sola_is_linear_operator(self, unit_box, space):
        """SOLAOperator3D must be an instance of pygeoinf.LinearOperator."""
        k = const_fn(unit_box, 1.0)
        m = VolumeKernel3D(space, k)
        codomain = EuclideanSpace(1)
        op = SOLAOperator3D(space, codomain, [m])
        assert isinstance(op, LinearOperator)


# ---------------------------------------------------------------------------
# Test: custom codomain — from_components controls output shape/type
# ---------------------------------------------------------------------------

class TestSOLACustomCodomainRealization:
    """The forward map delegates output construction.

    Output construction goes through ``codomain.from_components``.
    """

    def test_sola_forward_respects_custom_codomain_realization(
        self, unit_box, space
    ):
        """
        A codomain whose ``from_components`` reverses the component vector
        must produce reversed output.

        Kernels on f(x) = x₀:
          m₁: kernel=1   → raw component 0.5
          m₂: kernel=2   → raw component 1.0

        Raw components [0.5, 1.0] → reversed by custom codomain → [1.0, 0.5].
        """
        # Lightweight test double for a dim-2 codomain with reversed
        # from_components.
        import pygeoinf

        class ReversedEuclidean(pygeoinf.HilbertSpace):
            """dim-2 Hilbert space with reversed ``from_components``."""

            @property
            def dim(self) -> int:
                return 2

            def to_components(self, x: np.ndarray) -> np.ndarray:
                return np.asarray(x, dtype=float)

            def from_components(self, c: np.ndarray) -> np.ndarray:
                return np.asarray(c, dtype=float)[::-1].copy()

            def to_dual(self, x):
                from pygeoinf.linear_forms import LinearForm
                return LinearForm(self, components=self.to_components(x))

            def from_dual(self, xp):
                return self.dual.to_components(xp)

            def inner_product(self, x1, x2) -> float:
                return float(np.dot(x1, x2))

            def __eq__(self, other):
                return isinstance(other, ReversedEuclidean)

        k1 = const_fn(unit_box, 1.0)
        k2 = const_fn(unit_box, 2.0)
        m1 = VolumeKernel3D(space, k1)
        m2 = VolumeKernel3D(space, k2)

        codomain = ReversedEuclidean()
        op = SOLAOperator3D(space, codomain, [m1, m2])

        f = linear_fn(unit_box, 0)   # f(x) = x₀; ∫ f dV = 0.5; ∫ 2f dV = 1.0
        y = op(f)

        # raw = [0.5, 1.0]; reversed = [1.0, 0.5]
        npt.assert_allclose(y[0], 1.0, rtol=5e-3, atol=5e-3)
        npt.assert_allclose(y[1], 0.5, rtol=5e-3, atol=5e-3)


# ---------------------------------------------------------------------------
# Test: validation — constructor must reject incorrect inputs
# ---------------------------------------------------------------------------

class TestSOLAValidation:

    def test_sola_rejects_mismatched_kernel_counts(self, unit_box, space):
        """
        3 kernels with a dim-2 codomain must raise ValueError.
        """
        kernels = [
            VolumeKernel3D(space, const_fn(unit_box, float(i)))
            for i in range(3)
        ]
        codomain = EuclideanSpace(2)

        with pytest.raises(ValueError, match="Number of kernels"):
            SOLAOperator3D(space, codomain, kernels)

    def test_sola_rejects_non_volume_kernel_kernel(self, unit_box, space):
        """
        A non-VolumeKernel3D item in the kernels list must raise
        TypeError.
        """
        vk = VolumeKernel3D(space, const_fn(unit_box, 1.0))
        codomain = EuclideanSpace(2)

        with pytest.raises(TypeError, match="VolumeKernel3D"):
            SOLAOperator3D(space, codomain, [vk, "not_a_kernel"])

    def test_sola_rejects_wrong_domain_kernel(self, unit_box):
        """
        A VolumeKernel3D from a different Lebesgue3D instance must raise
        ValueError.
        """
        space_a = Lebesgue3D(unit_box)
        space_b = Lebesgue3D(unit_box)

        m_on_b = VolumeKernel3D(space_b, const_fn(unit_box, 1.0))
        codomain = EuclideanSpace(1)

        with pytest.raises(ValueError, match="same space instance"):
            SOLAOperator3D(space_a, codomain, [m_on_b])

    def test_sola_rejects_non_lebesgue_domain(self):
        """
        Passing a non-Lebesgue3D domain must raise TypeError.
        """
        codomain = EuclideanSpace(2)
        fake_domain = EuclideanSpace(100)

        with pytest.raises(TypeError, match="Lebesgue3D"):
            SOLAOperator3D(fake_domain, codomain, [])

    def test_sola_rejects_zero_kernels(self, space):
        """
        An empty kernels list must raise ValueError at construction time,
        guarding against an IndexError in _dual_mapping on empty coordinates.
        The guard fires before the len != codomain.dim check.
        """
        with pytest.raises(ValueError, match="at least one kernel"):
            SOLAOperator3D(space, EuclideanSpace(1), [])

    def test_sola_kernels_property_returns_copy(self, unit_box, space):
        """
        The .kernels property returns a copy; external mutation is safe.
        """
        vk = VolumeKernel3D(space, const_fn(unit_box, 1.0))
        codomain = EuclideanSpace(1)
        op = SOLAOperator3D(space, codomain, [vk])

        kernels = op.kernels
        kernels.clear()
        assert len(op.kernels) == 1


# ---------------------------------------------------------------------------
# Phase 2 tests: dual and adjoint mappings
# ---------------------------------------------------------------------------

def _make_mass_weighted_codomain(underlying_space, mass_diag):
    """Helper: build a MassWeightedHilbertSpace with diagonal mass matrix."""
    M = np.diag(mass_diag)
    M_inv = np.diag(1.0 / np.asarray(mass_diag))
    mass_op = LinearOperator.from_matrix(underlying_space, underlying_space, M)
    inv_mass_op = LinearOperator.from_matrix(
        underlying_space,
        underlying_space,
        M_inv,
    )
    return MassWeightedHilbertSpace(underlying_space, mass_op, inv_mass_op)


class TestSOLADualMapping:
    """Dual mapping extracts codomain-dual coordinates.

    The result is the weighted sum of the kernel-induced forms.
    """

    def test_sola_dual_uses_codomain_dual_coordinates(self, unit_box, space):
        """
        For kernels k₁ = 1 and k₂ = x₀, and a dual element
        yp with components [3, 5], the dual result xp must satisfy:

            xp(g) = 3 * m₁(g) + 5 * m₂(g)

        for an arbitrary test function g = x₀ (linear in first coordinate).

        Analytic values on [0,1]³:
            m₁(x₀) = ∫ 1 · x₀ dV  = 0.5
            m₂(x₀) = ∫ x₀ · x₀ dV = 1/3
            xp(x₀) = 3·0.5 + 5·(1/3) = 1.5 + 5/3 ≈ 3.167
        """
        k1 = const_fn(unit_box, 1.0)
        k2 = linear_fn(unit_box, 0)
        m1 = VolumeKernel3D(space, k1)
        m2 = VolumeKernel3D(space, k2)

        codomain = EuclideanSpace(2)
        op = SOLAOperator3D(space, codomain, [m1, m2])

        # A codomain-dual element with components [3, 5]
        yp = LinearForm(codomain, components=np.array([3.0, 5.0]))
        xp = op.dual(yp)

        g = linear_fn(unit_box, 0)  # g(x) = x₀
        expected = 3.0 * m1(g) + 5.0 * m2(g)
        npt.assert_allclose(xp(g), expected, rtol=5e-3, atol=5e-3)

    def test_sola_dual_single_measurement(self, unit_box, space):
        """
        Single measurement: dual with yp = [c] gives xp = c * m₁.
        """
        k1 = linear_fn(unit_box, 2)
        m1 = VolumeKernel3D(space, k1)
        codomain = EuclideanSpace(1)
        op = SOLAOperator3D(space, codomain, [m1])

        c = 7.0
        yp = LinearForm(codomain, components=np.array([c]))
        xp = op.dual(yp)

        g = poly_fn(unit_box, 2, 2)   # g(x) = x₂²
        npt.assert_allclose(xp(g), c * m1(g), rtol=5e-3, atol=5e-3)


class TestSOLAAdjoints:
    """Adjoint identity holds for both codomain types."""

    def test_sola_adjoint_matches_lebesgue_pairing(self, unit_box, space):
        """
        EuclideanSpace codomain: adjoint identity for y = [2, -1], f = x₀.

        Analytic:
            Af = [m₁(f), m₂(f)] = [0.5, 1/3]   (m₁ kernel=1, m₂ kernel=x₀)
            ⟨y, Af⟩ = [2, -1]·[0.5, 1/3] = 1.0 - 1/3 = 2/3
            A†y kernel = 2·1 + (−1)·x₀ = 2 − x₀
            ⟨A†y, f⟩ = ∫ (2 − x₀)·x₀ dV = 2·0.5 − 1/3 = 2/3  ✓
        """
        k1 = const_fn(unit_box, 1.0)
        k2 = linear_fn(unit_box, 0)
        m1 = VolumeKernel3D(space, k1)
        m2 = VolumeKernel3D(space, k2)

        codomain = EuclideanSpace(2)
        op = SOLAOperator3D(space, codomain, [m1, m2])

        y = np.array([2.0, -1.0])
        f = linear_fn(unit_box, 0)   # f(x) = x₀

        lhs = space.inner_product(op.adjoint(y), f)
        rhs = codomain.inner_product(y, op(f))
        npt.assert_allclose(lhs, rhs, rtol=5e-3, atol=5e-3)

    def test_sola_mass_weighted_codomain_adjoint_identity(
        self, unit_box, space
    ):
        """
        MassWeightedHilbertSpace codomain with M = diag([2, 3]):
        adjoint identity ⟨A†y, f⟩_V = ⟨y, Af⟩_W for y = [1, 1], f = x₀.

        Analytic:
            Af = [0.5, 1/3]
            ⟨y, Af⟩_W = (M·y)·(Af) = [2,3]·[0.5,1/3] = 1.0 + 1.0 = 2.0
            A†y kernel = 2·1 + 3·x₀
            ⟨A†y, f⟩_V = ∫ (2+3x₀)·x₀ dV = 2·0.5 + 3·(1/3) = 1.0 + 1.0 = 2.0  ✓
        """
        k1 = const_fn(unit_box, 1.0)
        k2 = linear_fn(unit_box, 0)
        m1 = VolumeKernel3D(space, k1)
        m2 = VolumeKernel3D(space, k2)

        underlying = EuclideanSpace(2)
        codomain = _make_mass_weighted_codomain(underlying, [2.0, 3.0])
        op = SOLAOperator3D(space, codomain, [m1, m2])

        y = np.array([1.0, 1.0])
        f = linear_fn(unit_box, 0)   # f(x) = x₀

        lhs = space.inner_product(op.adjoint(y), f)
        rhs = codomain.inner_product(y, op(f))
        npt.assert_allclose(lhs, rhs, rtol=5e-3, atol=5e-3)


# ---------------------------------------------------------------------------
# Phase 3 tests: provider-backed lazy kernel access
# ---------------------------------------------------------------------------

class TestSOLAProviderBacked:
    """Phase 3: SOLAOperator3D accepts a KernelProvider3D for lazy access."""

    def test_sola_accepts_volume_kernel_provider(self, unit_box, space):
        """SOLAOperator3D can be constructed with a FunctionToKernelAdapter.

        The operator must be a LinearOperator instance with the correct
        codomain dimension.
        """
        from pygeoinf3d.providers.base import FunctionToKernelAdapter
        from pygeoinf3d.providers.tensor_product import (
            BoxLaplacianSpectrumProvider,
        )

        provider = BoxLaplacianSpectrumProvider(
            unit_box, truncation=2, bc_type='dirichlet'
        )
        adapter = FunctionToKernelAdapter(provider, space)
        codomain = EuclideanSpace(adapter.dim)

        op = SOLAOperator3D(space, codomain, adapter)

        assert isinstance(op, LinearOperator)
        assert op.codomain.dim == adapter.dim

    def test_sola_lazy_provider_access_matches_list_kernels(
        self, unit_box, space
    ):
        """Provider-backed forward map equals sequence-backed forward map.

        Build two operators — one with an explicit kernel list, one with
        a FunctionToKernelAdapter — and verify they produce the same output
        for a constant test function.
        """
        from pygeoinf3d.providers.base import FunctionToKernelAdapter
        from pygeoinf3d.providers.tensor_product import BoxLaplacianSpectrumProvider

        provider = BoxLaplacianSpectrumProvider(
            unit_box, truncation=2, bc_type='dirichlet'
        )
        m = provider.dim  # 8 modes
        adapter = FunctionToKernelAdapter(provider, space)
        codomain = EuclideanSpace(m)

        # Sequence-backed operator
        kernels_list = [
            VolumeKernel3D(space, provider.get_eigenfunction(i))
            for i in range(m)
        ]
        op_seq = SOLAOperator3D(space, codomain, kernels_list)

        # Provider-backed operator
        op_prov = SOLAOperator3D(space, codomain, adapter)

        f = const_fn(unit_box, 1.0)
        y_seq = op_seq(f)
        y_prov = op_prov(f)

        npt.assert_allclose(y_prov, y_seq, rtol=1e-10, atol=1e-10)

    def test_sola_provider_cache_returns_stable_kernels(
        self, unit_box, space
    ):
        """Repeated _get_kernel(i) calls return the same VolumeKernel3D object.

        The cache must ensure that the provider's get_kernel is not called
        more than once per index, and the returned objects are identity-equal.
        """
        from pygeoinf3d.providers.base import FunctionToKernelAdapter
        from pygeoinf3d.providers.tensor_product import BoxLaplacianSpectrumProvider

        provider = BoxLaplacianSpectrumProvider(
            unit_box, truncation=2, bc_type='dirichlet'
        )
        adapter = FunctionToKernelAdapter(provider, space)
        codomain = EuclideanSpace(provider.dim)
        op = SOLAOperator3D(space, codomain, adapter)

        # First and second access to the same index must be identity-equal
        k0_first = op._get_kernel(0)
        k0_second = op._get_kernel(0)
        assert k0_first is k0_second, (
            "_get_kernel(0) returned different objects on repeated calls; "
            "cache is not working."
        )

        # Access through .kernels must also use the cache
        ks1 = op.kernels
        ks2 = op.kernels
        assert len(ks1) == provider.dim
        for vk in ks1:
            assert isinstance(vk, VolumeKernel3D)
        # Same cached objects across two .kernels calls
        for kk1, kk2 in zip(ks1, ks2):
            assert kk1 is kk2, (
                ".kernels did not return identity-equal elements on "
                "repeated calls."
            )

    def test_function_to_kernel_adapter_uses_generic_get_function(
        self, unit_box, space
    ):
        """The adapter depends on generic function-provider naming.

        A provider with ``get_function`` but no ``get_eigenfunction`` must
        still work.
        """
        from pygeoinf3d.providers.base import FunctionToKernelAdapter

        class ConstantFunctionProvider:
            @property
            def dim(self) -> int:
                return 1

            def get_function(self, i: int) -> Function3D:
                if i != 0:
                    raise IndexError(i)
                return const_fn(unit_box, 2.0)

        provider = ConstantFunctionProvider()
        adapter = FunctionToKernelAdapter(provider, space)
        kernel = adapter.get_kernel(0)

        test_function = const_fn(unit_box, 3.0)
        npt.assert_allclose(kernel(test_function), 6.0, rtol=1e-10, atol=1e-10)


# ---------------------------------------------------------------------------
# Phase 4 tests: public export surface lock
# ---------------------------------------------------------------------------

class TestPhase4PublicAPIExports:
    """Phase 4: public export surface is locked via import-level assertions.

    These tests verify that all SOLA symbols are re-exported at the
    expected package boundaries.  Any accidental removal from __all__ or
    the top-level namespace will cause exactly these tests to fail first,
    making regressions in the public API immediately visible.

    Extension-boundary note
    -----------------------
    The forward and dual APIs are representation-independent: they work
    for any VolumeKernel3D sequence or KernelProvider3D without touching
    domain coefficients.  The adjoint mapping is narrower: it relies on
    ``domain.from_dual(xp)`` to realise the dual element as a primal
    Function3D, and ``Lebesgue3D.from_dual`` currently only accepts
    VolumeKernel3D instances owned by that same space instance.  Because
    ``SOLAOperator3D.dual()`` always returns a VolumeKernel3D, the
    round-trip holds — but only due to this alignment.  Any future
    operator subclass that changes the dual-mapping return type must
    either extend ``from_dual`` or override ``adjoint`` accordingly.
    """

    def test_top_level_exports_sola_operator(self):
        """pygeoinf3d.SOLAOperator3D exists and is the canonical class."""
        import pygeoinf3d
        assert hasattr(pygeoinf3d, "SOLAOperator3D")
        assert pygeoinf3d.SOLAOperator3D is SOLAOperator3D

    def test_top_level_exports_function_provider(self):
        """pygeoinf3d.FunctionProvider3D is importable at the top level."""
        import pygeoinf3d
        from pygeoinf3d.providers.base import FunctionProvider3D
        assert hasattr(pygeoinf3d, "FunctionProvider3D")
        assert pygeoinf3d.FunctionProvider3D is FunctionProvider3D

    def test_top_level_exports_kernel_provider(self):
        """pygeoinf3d.KernelProvider3D is importable at the top level."""
        import pygeoinf3d
        from pygeoinf3d.providers.base import KernelProvider3D
        assert hasattr(pygeoinf3d, "KernelProvider3D")
        assert pygeoinf3d.KernelProvider3D is KernelProvider3D

    def test_top_level_exports_function_to_kernel_adapter(self):
        """FunctionToKernelAdapter is importable at the top level."""
        import pygeoinf3d
        from pygeoinf3d.providers.base import FunctionToKernelAdapter
        assert hasattr(pygeoinf3d, "FunctionToKernelAdapter")
        assert pygeoinf3d.FunctionToKernelAdapter is FunctionToKernelAdapter

    def test_operators_subpackage_exports_sola_operator(self):
        """pygeoinf3d.operators.SOLAOperator3D is the same class object."""
        from pygeoinf3d.operators import SOLAOperator3D as SOLAFromOperators
        assert SOLAFromOperators is SOLAOperator3D

    def test_operators_dunder_all_contains_sola_operator(self):
        """SOLAOperator3D appears in pygeoinf3d.operators.__all__."""
        import pygeoinf3d.operators
        assert "SOLAOperator3D" in pygeoinf3d.operators.__all__

    def test_sola_operator_in_dunder_all(self):
        """SOLAOperator3D appears in pygeoinf3d.__all__."""
        import pygeoinf3d
        assert "SOLAOperator3D" in pygeoinf3d.__all__

    def test_function_provider_in_dunder_all(self):
        """FunctionProvider3D appears in pygeoinf3d.__all__."""
        import pygeoinf3d
        assert "FunctionProvider3D" in pygeoinf3d.__all__

    def test_kernel_provider_in_dunder_all(self):
        """KernelProvider3D appears in pygeoinf3d.__all__."""
        import pygeoinf3d
        assert "KernelProvider3D" in pygeoinf3d.__all__

    def test_function_to_kernel_adapter_in_dunder_all(self):
        """FunctionToKernelAdapter appears in pygeoinf3d.__all__."""
        import pygeoinf3d
        assert "FunctionToKernelAdapter" in pygeoinf3d.__all__


# ---------------------------------------------------------------------------
# Phase 5 tests: PathIntegralFunctional as SOLA kernels
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Helper for TestSOLAOperator3DWithPathFunctionals
# ---------------------------------------------------------------------------

def _make_sobolev_from_lebesgue(space: Lebesgue3D, alpha: float = 2.0) -> Sobolev3D:
    """Wrap *space* in a Sobolev3D with a simple scalar-multiple weight.

    W = alpha * I,  W^{-1} = (1/alpha) * I.  Not spectral.
    """
    a = float(alpha)
    inv_a = 1.0 / a

    def _scale(f: Function3D) -> Function3D:
        return Function3D(f.domain, evaluate_callable=lambda x: a * f(x))

    def _inv_scale(f: Function3D) -> Function3D:
        return Function3D(f.domain, evaluate_callable=lambda x: inv_a * f(x))

    weight = LinearOperator(space, space, _scale, adjoint_mapping=_scale)
    inv_weight = LinearOperator(space, space, _inv_scale, adjoint_mapping=_inv_scale)
    return Sobolev3D(space, weight, inv_weight)


class TestSOLAOperator3DWithPathFunctionals:
    """SOLAOperator3D accepts PathIntegralFunctional as measurement forms.

    Covers (Phase 3 domain contracts):
    - Construction with a list of PathIntegralFunctional on Sobolev3D succeeds
    - Forward map with constant function: A(f)[i] = c * arclength_i
    - Dual of a unit codomain dual gives a scalar multiple of PathIntegralFunctional
    - Adjoint raises TypeError (SpectralOperator3D not available) for non-spectral weight
    - Mixing VolumeKernel3D and PathIntegralFunctional on the same Sobolev3D is accepted
    - Backward compatibility: VolumeKernel3D workflows on Lebesgue3D are unchanged
    """

    @pytest.fixture
    def sobolev_space(self, space):
        """Sobolev3D wrapping the module-level Lebesgue3D space fixture."""
        return _make_sobolev_from_lebesgue(space)

    def test_accepts_path_integral_functionals(self, unit_box, sobolev_space):
        """SOLAOperator3D construction with PathIntegralFunctional list succeeds."""
        from pygeoinf3d.spaces.forms import PathIntegralFunctional
        from pygeoinf3d.core.paths import Path3D

        path_a = Path3D.from_polyline(
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        )
        path_b = Path3D.from_polyline(
            np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        )
        form_a = PathIntegralFunctional(sobolev_space, path_a)
        form_b = PathIntegralFunctional(sobolev_space, path_b)

        codomain = EuclideanSpace(2)
        op = SOLAOperator3D(sobolev_space, codomain, [form_a, form_b])

        assert isinstance(op, LinearOperator)
        assert op.codomain.dim == 2

    def test_forward_path_integral_constant_fn(self, unit_box, sobolev_space):
        """Forward map with constant f=c gives c * arclength for each path.

        Path a: (0,0,0)→(1,0,0), arclength = 1.0  → ℓ_a(c) = c * 1.0
        Path b: (0,0,0)→(0,1,0), arclength = 1.0  → ℓ_b(c) = c * 1.0
        """
        from pygeoinf3d.spaces.forms import PathIntegralFunctional
        from pygeoinf3d.core.paths import Path3D

        path_a = Path3D.from_polyline(
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        )
        path_b = Path3D.from_polyline(
            np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        )
        form_a = PathIntegralFunctional(sobolev_space, path_a)
        form_b = PathIntegralFunctional(sobolev_space, path_b)

        c = 3.0
        codomain = EuclideanSpace(2)
        op = SOLAOperator3D(sobolev_space, codomain, [form_a, form_b])

        f = const_fn(unit_box, c)
        y = op(f)

        npt.assert_allclose(y[0], c * path_a.arclength, rtol=1e-3, atol=1e-3)
        npt.assert_allclose(y[1], c * path_b.arclength, rtol=1e-3, atol=1e-3)

    def test_dual_mapping_path_functionals(self, unit_box, sobolev_space):
        """Dual of a unit codomain dual returns a scalar PathIntegralFunctional.

        With a single PathIntegralFunctional and yp with component [c],
        op.dual(yp) must return a PathIntegralFunctional that evaluates as
        c * form(f) for a test function f.
        """
        from pygeoinf3d.spaces.forms import PathIntegralFunctional
        from pygeoinf3d.core.paths import Path3D

        path = Path3D.from_polyline(
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        )
        form = PathIntegralFunctional(sobolev_space, path)

        codomain = EuclideanSpace(1)
        op = SOLAOperator3D(sobolev_space, codomain, [form])

        c = 2.5
        yp = LinearForm(codomain, components=np.array([c]))
        xp = op.dual(yp)

        assert isinstance(xp, PathIntegralFunctional)

        # Verify: xp(f) = c * form(f) for a constant test function
        f = const_fn(unit_box, 1.0)
        npt.assert_allclose(xp(f), c * form(f), rtol=1e-6, atol=1e-6)

    def test_dual_mapping_multi_path_returns_sum_form(self, unit_box, sobolev_space):
        """Dual of a 2-path stack returns a SumLinearForm."""
        from pygeoinf3d.spaces.forms import PathIntegralFunctional, SumLinearForm
        from pygeoinf3d.core.paths import Path3D

        path_a = Path3D.from_polyline(
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        )
        path_b = Path3D.from_polyline(
            np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        )
        form_a = PathIntegralFunctional(sobolev_space, path_a)
        form_b = PathIntegralFunctional(sobolev_space, path_b)

        codomain = EuclideanSpace(2)
        op = SOLAOperator3D(sobolev_space, codomain, [form_a, form_b])

        yp = LinearForm(codomain, components=np.array([1.0, 2.0]))
        xp = op.dual(yp)
        assert isinstance(xp, SumLinearForm)
        # Verify evaluation: xp(f) = 1*form_a(f) + 2*form_b(f)
        f = const_fn(unit_box, 1.0)
        expected = 1.0 * form_a(f) + 2.0 * form_b(f)
        npt.assert_allclose(xp(f), expected, rtol=1e-6, atol=1e-6)

    def test_adjoint_path_functionals_raises_clear_error(self, unit_box, sobolev_space):
        """Adjoint with path forms on a non-spectral Sobolev3D raises TypeError.

        The Sobolev3D fixture uses a plain scaling weight (not SpectralOperator3D).
        Sobolev3D.from_dual Branch 3 requires SpectralOperator3D for path-form
        representers, so the adjoint raises TypeError with a clear message.
        """
        from pygeoinf3d.spaces.forms import PathIntegralFunctional
        from pygeoinf3d.core.paths import Path3D

        path_a = Path3D.from_polyline(
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        )
        path_b = Path3D.from_polyline(
            np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        )
        form_a = PathIntegralFunctional(sobolev_space, path_a)
        form_b = PathIntegralFunctional(sobolev_space, path_b)

        codomain = EuclideanSpace(2)
        op = SOLAOperator3D(sobolev_space, codomain, [form_a, form_b])

        y = np.array([1.0, 1.0])
        with pytest.raises(TypeError, match="SpectralOperator3D"):
            op.adjoint(y)

    def test_mixed_form_types_accepted(self, unit_box, sobolev_space):
        """Mixing VolumeKernel3D and PathIntegralFunctional on the same
        Sobolev3D is accepted.

        Both forms must have ``domain is sobolev_space``.  Construction must
        succeed and the forward map must evaluate each form correctly:
        VolumeKernel3D gives a volume integral, PathIntegralFunctional gives
        a path integral.
        """
        from pygeoinf3d.spaces.forms import PathIntegralFunctional
        from pygeoinf3d.core.paths import Path3D

        vk = VolumeKernel3D(sobolev_space, const_fn(unit_box, 1.0))
        path = Path3D.from_polyline(
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        )
        pif = PathIntegralFunctional(sobolev_space, path)

        codomain = EuclideanSpace(2)
        # Construction must not raise
        op = SOLAOperator3D(sobolev_space, codomain, [vk, pif])
        assert isinstance(op, LinearOperator)

        # Forward: constant f=2.0
        # vk(f) = ∫ 1·2 dV = 2.0  (unit box has volume 1)
        # pif(f) = 2.0 * arclength = 2.0 * 1.0 = 2.0
        f = const_fn(unit_box, 2.0)
        y = op(f)
        npt.assert_allclose(y[0], 2.0, rtol=5e-3, atol=5e-3)
        npt.assert_allclose(y[1], 2.0 * path.arclength, rtol=1e-3, atol=1e-3)

    def test_backward_compat_volume_kernels_still_work(self, unit_box, space):
        """Phase 5 generalization does not break existing VolumeKernel3D workflow.

        Checks forward + adjoint identity identical to Phase 2 analytic tests.
        """
        k1 = const_fn(unit_box, 1.0)
        k2 = linear_fn(unit_box, 0)
        m1 = VolumeKernel3D(space, k1)
        m2 = VolumeKernel3D(space, k2)

        codomain = EuclideanSpace(2)
        op = SOLAOperator3D(space, codomain, [m1, m2])

        # Forward: ∫ 1·x₀ dV = 0.5, ∫ x₀·x₀ dV = 1/3
        f = linear_fn(unit_box, 0)
        y = op(f)
        npt.assert_allclose(y[0], 0.5,       rtol=5e-3, atol=5e-3)
        npt.assert_allclose(y[1], 1.0 / 3.0, rtol=5e-3, atol=5e-3)

        # Adjoint identity ⟨A†y, f⟩_V = ⟨y, Af⟩_W
        y_vec = np.array([2.0, -1.0])
        lhs = space.inner_product(op.adjoint(y_vec), f)
        rhs = codomain.inner_product(y_vec, op(f))
        npt.assert_allclose(lhs, rhs, rtol=5e-3, atol=5e-3)


# ---------------------------------------------------------------------------
# build_path_cache fast-path tests
# ---------------------------------------------------------------------------

class TestSOLABuildPathCache:
    """Tests for SOLAOperator3D.build_path_cache().

    Covers:
    - Correctness: cached forward matches uncached to machine precision.
    - Flag: _path_cache_built is True after calling build_path_cache().
    - TypeError: calling build_path_cache() when any form is not PathIntegralFunctional.
    - ValueError: calling build_path_cache() when forms have different n_quad values.
    """

    @pytest.fixture
    def sobolev_space(self, unit_box, space):
        return _make_sobolev_from_lebesgue(space)

    def _make_path_forms(self, unit_box, sobolev_space, N=4, n_quad=50):
        """Return N PathIntegralFunctional forms on *sobolev_space*."""
        from pygeoinf3d.spaces.forms import PathIntegralFunctional
        from pygeoinf3d.core.paths import Path3D

        rng = np.random.default_rng(42)
        srcs = rng.uniform(0.1, 0.9, size=(N, 3))
        stas = rng.uniform(0.1, 0.9, size=(N, 3))
        stas[:, 2] = 0.95  # keep stations near top face
        paths = [
            Path3D.from_polyline(np.array([srcs[i], stas[i]]))
            for i in range(N)
        ]
        return [PathIntegralFunctional(sobolev_space, p, n_quad=n_quad) for p in paths]

    def test_path_cache_correctness(self, unit_box, space, sobolev_space):
        """Cached forward evaluation matches uncached to machine precision.

        For N=6 path forms, build_path_cache() output must agree with the
        sequential (uncached) forward pass to rtol=1e-10, atol=1e-14.
        """
        forms = self._make_path_forms(unit_box, sobolev_space, N=6, n_quad=80)
        f = Function3D(
            unit_box,
            evaluate_callable=lambda x: np.exp(
                -2.0 * np.sum(np.asarray(x) ** 2, axis=-1)
            ),
        )

        codomain = EuclideanSpace(6)
        G_slow = SOLAOperator3D(sobolev_space, codomain, forms)
        G_fast = SOLAOperator3D(sobolev_space, codomain, forms)
        G_fast.build_path_cache()

        y_ref = G_slow(f)
        y_cached = G_fast(f)

        npt.assert_allclose(y_cached, y_ref, rtol=1e-10, atol=1e-14)

    def test_path_cache_built_flag_set(self, unit_box, space, sobolev_space):
        """_path_cache_built is False before and True after build_path_cache()."""
        forms = self._make_path_forms(unit_box, sobolev_space, N=3)
        codomain = EuclideanSpace(3)
        op = SOLAOperator3D(sobolev_space, codomain, forms)

        assert op._path_cache_built is False
        op.build_path_cache()
        assert op._path_cache_built is True

    def test_path_cache_raises_type_error_for_volume_kernel(self, unit_box, space, sobolev_space):
        """build_path_cache() raises TypeError when any form is a VolumeKernel3D."""
        from pygeoinf3d.spaces.forms import PathIntegralFunctional
        from pygeoinf3d.core.paths import Path3D

        path = Path3D.from_polyline(
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        )
        pif = PathIntegralFunctional(sobolev_space, path, n_quad=50)
        vk = VolumeKernel3D(sobolev_space, const_fn(unit_box, 1.0))

        codomain = EuclideanSpace(2)
        op = SOLAOperator3D(sobolev_space, codomain, [pif, vk])

        with pytest.raises(TypeError, match="PathIntegralFunctional"):
            op.build_path_cache()

    def test_path_cache_raises_value_error_for_mixed_n_quad(self, unit_box, space, sobolev_space):
        """build_path_cache() raises ValueError when forms have different n_quad."""
        from pygeoinf3d.spaces.forms import PathIntegralFunctional
        from pygeoinf3d.core.paths import Path3D

        path_a = Path3D.from_polyline(
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        )
        path_b = Path3D.from_polyline(
            np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        )
        form_a = PathIntegralFunctional(sobolev_space, path_a, n_quad=50)
        form_b = PathIntegralFunctional(sobolev_space, path_b, n_quad=100)

        codomain = EuclideanSpace(2)
        op = SOLAOperator3D(sobolev_space, codomain, [form_a, form_b])

        with pytest.raises(ValueError, match="n_quad"):
            op.build_path_cache()


# ---------------------------------------------------------------------------
# build_mixed_cache fast-path tests
# ---------------------------------------------------------------------------

class TestSOLABuildMixedCache:
    """Tests for SOLAOperator3D.build_mixed_cache().

    Covers:
    - Correctness for a truly mixed stack (volume + path forms together).
    - Works for all-volume stacks (same result as build_vol_cache).
    - Works for all-path stacks (same result as build_path_cache).
    - Flag: _mixed_cache_built is True after calling build_mixed_cache().
    - TypeError: unknown form type in the stack.
    - ValueError: path forms with non-uniform n_quad.
    """

    @pytest.fixture
    def sobolev_space(self, unit_box, space):
        return _make_sobolev_from_lebesgue(space)

    def _make_vol_forms(self, unit_box, sobolev_space, N=3):
        """Return N VolumeKernel3D forms on *sobolev_space*."""
        centers = np.linspace(0.2, 0.8, N)
        forms = []
        for cx in centers:
            c = float(cx)
            forms.append(VolumeKernel3D(
                sobolev_space,
                Function3D(
                    unit_box,
                    evaluate_callable=lambda x, cx=c: np.exp(
                        -20.0 * np.sum((np.asarray(x) - cx) ** 2, axis=-1)
                    ),
                ),
            ))
        return forms

    def _make_path_forms(self, unit_box, sobolev_space, N=3, n_quad=60):
        """Return N PathIntegralFunctional forms on *sobolev_space*."""
        from pygeoinf3d.spaces.forms import PathIntegralFunctional
        from pygeoinf3d.core.paths import Path3D

        rng = np.random.default_rng(7)
        srcs = rng.uniform(0.1, 0.85, size=(N, 3))
        stas = rng.uniform(0.1, 0.85, size=(N, 3))
        paths = [
            Path3D.from_polyline(np.array([srcs[i], stas[i]]))
            for i in range(N)
        ]
        return [PathIntegralFunctional(sobolev_space, p, n_quad=n_quad) for p in paths]

    def test_mixed_cache_correctness(self, unit_box, space, sobolev_space):
        """Cached forward matches uncached for a mixed vol+path stack.

        Uses 2 VolumeKernel3D + 3 PathIntegralFunctional forms interleaved
        in original-order.  Results must agree to rtol=1e-10, atol=1e-14.
        """
        vol_forms = self._make_vol_forms(unit_box, sobolev_space, N=2)
        path_forms = self._make_path_forms(unit_box, sobolev_space, N=3, n_quad=60)
        # Interleave: vol[0], path[0], vol[1], path[1], path[2]
        forms = [vol_forms[0], path_forms[0], vol_forms[1], path_forms[1], path_forms[2]]

        f = Function3D(
            unit_box,
            evaluate_callable=lambda x: np.exp(
                -3.0 * np.sum(np.asarray(x) ** 2, axis=-1)
            ),
        )
        codomain = EuclideanSpace(5)
        G_slow = SOLAOperator3D(sobolev_space, codomain, forms)
        G_fast = SOLAOperator3D(sobolev_space, codomain, forms)
        G_fast.build_mixed_cache(resolution=20)

        npt.assert_allclose(G_fast(f), G_slow(f), rtol=1e-10, atol=1e-14)

    def test_mixed_cache_flag_set(self, unit_box, space, sobolev_space):
        """_mixed_cache_built starts False and is True after build_mixed_cache()."""
        forms = self._make_vol_forms(unit_box, sobolev_space, N=2) + \
                self._make_path_forms(unit_box, sobolev_space, N=2)
        op = SOLAOperator3D(sobolev_space, EuclideanSpace(4), forms)
        assert op._mixed_cache_built is False
        op.build_mixed_cache()
        assert op._mixed_cache_built is True

    def test_mixed_cache_all_vol(self, unit_box, space, sobolev_space):
        """build_mixed_cache works for all-volume stacks."""
        forms = self._make_vol_forms(unit_box, sobolev_space, N=4)
        f = Function3D(unit_box, evaluate_callable=lambda x: np.ones(len(x)))
        codomain = EuclideanSpace(4)
        G_slow = SOLAOperator3D(sobolev_space, codomain, forms)
        G_fast = SOLAOperator3D(sobolev_space, codomain, forms)
        G_fast.build_mixed_cache(resolution=20)
        npt.assert_allclose(G_fast(f), G_slow(f), rtol=1e-8, atol=1e-12)

    def test_mixed_cache_all_path(self, unit_box, space, sobolev_space):
        """build_mixed_cache works for all-path stacks."""
        forms = self._make_path_forms(unit_box, sobolev_space, N=4)
        f = Function3D(
            unit_box,
            evaluate_callable=lambda x: np.exp(
                -np.sum(np.asarray(x) ** 2, axis=-1)
            ),
        )
        codomain = EuclideanSpace(4)
        G_slow = SOLAOperator3D(sobolev_space, codomain, forms)
        G_fast = SOLAOperator3D(sobolev_space, codomain, forms)
        G_fast.build_mixed_cache()
        npt.assert_allclose(G_fast(f), G_slow(f), rtol=1e-10, atol=1e-14)

    def test_mixed_cache_raises_type_error_unknown_form(self, unit_box, space, sobolev_space):
        """build_mixed_cache raises TypeError for any unsupported form type."""
        from pygeoinf.linear_forms import LinearForm

        class BogusForm(LinearForm):
            def __init__(self, domain):
                super().__init__(domain, mapping=lambda f: 0.0)

        vol = self._make_vol_forms(unit_box, sobolev_space, N=1)[0]
        bogus = BogusForm(sobolev_space)
        op = SOLAOperator3D(sobolev_space, EuclideanSpace(2), [vol, bogus])
        with pytest.raises(TypeError, match="build_mixed_cache"):
            op.build_mixed_cache()

    def test_mixed_cache_raises_value_error_nonuniform_path_n_quad(
        self, unit_box, space, sobolev_space
    ):
        """build_mixed_cache raises ValueError when path forms have different n_quad."""
        from pygeoinf3d.spaces.forms import PathIntegralFunctional
        from pygeoinf3d.core.paths import Path3D

        path_a = Path3D.from_polyline(np.array([[0.1, 0.1, 0.1], [0.9, 0.1, 0.1]]))
        path_b = Path3D.from_polyline(np.array([[0.1, 0.5, 0.1], [0.9, 0.5, 0.1]]))
        vol = self._make_vol_forms(unit_box, sobolev_space, N=1)[0]
        pif_a = PathIntegralFunctional(sobolev_space, path_a, n_quad=40)
        pif_b = PathIntegralFunctional(sobolev_space, path_b, n_quad=80)

        op = SOLAOperator3D(sobolev_space, EuclideanSpace(3), [vol, pif_a, pif_b])
        with pytest.raises(ValueError, match="n_quad"):
            op.build_mixed_cache()


# ---------------------------------------------------------------------------
# Fast path for G @ G.adjoint
# ---------------------------------------------------------------------------

class TestSOLAFastGramComposition:
    """Tests for the specialized fast path in ``G @ G.adjoint``.

    Covers:
    - All-volume Lebesgue stacks on BoxDomain.
    - Mixed volume/path spectral Sobolev stacks.
    - Strong non-activation when the prerequisites are not satisfied.
    """

    def test_gram_composition_volume_fast_path_bypasses_from_dual(
        self, unit_box, space, monkeypatch
    ):
        """All-volume BoxDomain case returns the Gram action without from_dual.

        If the fast path were not used, generic ``G @ G.adjoint`` would call
        ``space.from_dual`` through ``G.adjoint``.  Monkeypatch it to raise and
        verify the composition still works.
        """
        forms = [
            VolumeKernel3D(space, const_fn(unit_box, 1.0)),
            VolumeKernel3D(space, linear_fn(unit_box, 0)),
        ]
        G = SOLAOperator3D(space, EuclideanSpace(2), forms)

        def _boom(_xp):
            raise RuntimeError("from_dual should not be called")

        monkeypatch.setattr(space, "from_dual", _boom)

        H = G @ G.adjoint
        y = np.array([2.0, -1.0])
        got = H(y)

        gram = np.array([
            [1.0, 0.5],
            [0.5, 1.0 / 3.0],
        ])
        expected = gram @ y
        npt.assert_allclose(got, expected, rtol=5e-3, atol=5e-3)

    def test_gram_composition_mixed_spectral_fast_path_bypasses_from_dual(
        self, unit_box, space, monkeypatch
    ):
        """Mixed volume/path spectral case routes to the spectral Gram backend."""
        from pygeoinf3d.providers.tensor_product import BoxLaplacianSpectrumProvider
        from pygeoinf3d.operators.bessel_sobolev_box import BesselSobolevBox
        from pygeoinf3d.spaces.forms import PathIntegralFunctional
        from pygeoinf3d.core.paths import Path3D
        from pygeoinf3d.solvers.spectral_gram import SpectralGramSolver

        provider = BoxLaplacianSpectrumProvider(
            unit_box, truncation=(4, 4, 4), bc_type="neumann"
        )
        W = BesselSobolevBox(space, provider, shift=1.0, order=2.0)
        W_inv = BesselSobolevBox(space, provider, shift=1.0, order=-2.0)
        sobolev = Sobolev3D(space, W, W_inv)

        path_a = Path3D.from_polyline(
            np.array([[0.1, 0.1, 0.1], [0.9, 0.2, 0.2]])
        )
        path_b = Path3D.from_polyline(
            np.array([[0.1, 0.8, 0.3], [0.8, 0.9, 0.7]])
        )
        forms = [
            VolumeKernel3D(sobolev, const_fn(unit_box, 1.0)),
            PathIntegralFunctional(sobolev, path_a, n_quad=60),
            VolumeKernel3D(sobolev, linear_fn(unit_box, 0)),
            PathIntegralFunctional(sobolev, path_b, n_quad=60),
        ]
        G = SOLAOperator3D(sobolev, EuclideanSpace(4), forms)

        def _boom(_xp):
            raise RuntimeError("from_dual should not be called")

        monkeypatch.setattr(sobolev, "from_dual", _boom)

        H = G @ G.adjoint
        y = np.array([1.0, -2.0, 0.5, 3.0])
        got = H(y)

        gram = SpectralGramSolver(sobolev, forms).gram_matrix
        expected = gram @ y
        npt.assert_allclose(got, expected, rtol=1e-8, atol=1e-12)

    def test_gram_composition_non_spectral_path_stack_does_not_fast_path(
        self, unit_box, space, monkeypatch
    ):
        """Non-spectral Sobolev path stacks must fall back to the generic route."""
        from pygeoinf3d.spaces.forms import PathIntegralFunctional
        from pygeoinf3d.core.paths import Path3D

        sobolev = _make_sobolev_from_lebesgue(space)
        path_a = Path3D.from_polyline(
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        )
        path_b = Path3D.from_polyline(
            np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        )
        forms = [
            PathIntegralFunctional(sobolev, path_a, n_quad=40),
            PathIntegralFunctional(sobolev, path_b, n_quad=40),
        ]
        G = SOLAOperator3D(sobolev, EuclideanSpace(2), forms)

        def _boom(_xp):
            raise RuntimeError("generic path reached")

        monkeypatch.setattr(sobolev, "from_dual", _boom)

        H = G @ G.adjoint
        with pytest.raises(RuntimeError, match="generic path reached"):
            H(np.array([1.0, 2.0]))
