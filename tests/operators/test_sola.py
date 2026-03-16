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

        with pytest.raises(ValueError, match="same Lebesgue3D instance"):
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
        from pygeoinf3d.providers.tensor_product import TensorProductProvider

        provider = TensorProductProvider(
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
        from pygeoinf3d.providers.tensor_product import TensorProductProvider

        provider = TensorProductProvider(
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
        from pygeoinf3d.providers.tensor_product import TensorProductProvider

        provider = TensorProductProvider(
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

class TestSOLAOperator3DWithPathFunctionals:
    """SOLAOperator3D accepts PathIntegralFunctional as measurement forms.

    Covers (Phase 5):
    - Construction with a list of PathIntegralFunctional succeeds
    - Forward map with constant function: A(f)[i] = c * arclength_i
    - Dual of a unit codomain dual gives a scalar multiple of PathIntegralFunctional
    - Adjoint raises TypeError with a Phase 5 / Sobolev3D hint message
    - Mixing VolumeKernel3D and PathIntegralFunctional in the same list is rejected
    - Backward compatibility: VolumeKernel3D workflows are unchanged
    """

    def test_accepts_path_integral_functionals(self, unit_box, space):
        """SOLAOperator3D construction with PathIntegralFunctional list succeeds."""
        from pygeoinf3d.spaces.forms import PathIntegralFunctional
        from pygeoinf3d.core.paths import Path3D

        path_a = Path3D.from_polyline(
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        )
        path_b = Path3D.from_polyline(
            np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        )
        form_a = PathIntegralFunctional(space, path_a)
        form_b = PathIntegralFunctional(space, path_b)

        codomain = EuclideanSpace(2)
        op = SOLAOperator3D(space, codomain, [form_a, form_b])

        assert isinstance(op, LinearOperator)
        assert op.codomain.dim == 2

    def test_forward_path_integral_constant_fn(self, unit_box, space):
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
        form_a = PathIntegralFunctional(space, path_a)
        form_b = PathIntegralFunctional(space, path_b)

        c = 3.0
        codomain = EuclideanSpace(2)
        op = SOLAOperator3D(space, codomain, [form_a, form_b])

        f = const_fn(unit_box, c)
        y = op(f)

        npt.assert_allclose(y[0], c * path_a.arclength, rtol=1e-3, atol=1e-3)
        npt.assert_allclose(y[1], c * path_b.arclength, rtol=1e-3, atol=1e-3)

    def test_dual_mapping_path_functionals(self, unit_box, space):
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
        form = PathIntegralFunctional(space, path)

        codomain = EuclideanSpace(1)
        op = SOLAOperator3D(space, codomain, [form])

        c = 2.5
        yp = LinearForm(codomain, components=np.array([c]))
        xp = op.dual(yp)

        assert isinstance(xp, PathIntegralFunctional)

        # Verify: xp(f) = c * form(f) for a constant test function
        f = const_fn(unit_box, 1.0)
        npt.assert_allclose(xp(f), c * form(f), rtol=1e-6, atol=1e-6)

    def test_adjoint_path_functionals_raises_clear_error(self, unit_box, space):
        """Adjoint with path-integral forms on a Lebesgue3D raises TypeError
        with a helpful message pointing to Phase 5 / Sobolev3D representers."""
        from pygeoinf3d.spaces.forms import PathIntegralFunctional
        from pygeoinf3d.core.paths import Path3D

        path = Path3D.from_polyline(
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        )
        form = PathIntegralFunctional(space, path)

        codomain = EuclideanSpace(1)
        op = SOLAOperator3D(space, codomain, [form])

        y = np.array([1.0])
        with pytest.raises(TypeError, match="Phase 5"):
            op.adjoint(y)

    def test_mixed_form_types_rejected(self, unit_box, space):
        """Mixing VolumeKernel3D and PathIntegralFunctional raises TypeError."""
        from pygeoinf3d.spaces.forms import PathIntegralFunctional
        from pygeoinf3d.core.paths import Path3D

        vk = VolumeKernel3D(space, const_fn(unit_box, 1.0))
        path = Path3D.from_polyline(
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        )
        pif = PathIntegralFunctional(space, path)

        codomain = EuclideanSpace(2)
        with pytest.raises(TypeError, match="same type"):
            SOLAOperator3D(space, codomain, [vk, pif])

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
