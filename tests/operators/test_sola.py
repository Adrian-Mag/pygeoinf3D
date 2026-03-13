"""Tests for pygeoinf3d.operators.sola.SOLAOperator3D (Phase 1 + 2).

Covers (Phase 1):
- Forward evaluation matches analytic integrals for known kernels
- Custom codomain realization via from_components is used
- Rejection of mismatched measurement counts
- Rejection of non-VolumeKernel3D measurements
- Rejection of measurements from a different Lebesgue3D instance
- Rejection of non-Lebesgue3D domain
- SOLAOperator3D is a LinearOperator instance

Covers (Phase 2):
- Dual mapping extracts codomain-dual coordinates and returns
    weighted sum of forms
- Adjoint identity holds for EuclideanSpace codomain (pairing test)
- Adjoint identity holds for MassWeightedHilbertSpace codomain
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
        Two measurements on f(x) = x₀ over [0,1]³:

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

        Measurements on f(x) = x₀:
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
        3 measurements with a dim-2 codomain must raise ValueError.
        """
        measurements = [
            VolumeKernel3D(space, const_fn(unit_box, float(i)))
            for i in range(3)
        ]
        codomain = EuclideanSpace(2)

        with pytest.raises(ValueError, match="Number of measurements"):
            SOLAOperator3D(space, codomain, measurements)

    def test_sola_rejects_non_volume_kernel_measurement(self, unit_box, space):
        """
        A non-VolumeKernel3D item in the measurements list must raise
        TypeError.
        """
        vk = VolumeKernel3D(space, const_fn(unit_box, 1.0))
        codomain = EuclideanSpace(2)

        with pytest.raises(TypeError, match="VolumeKernel3D"):
            SOLAOperator3D(space, codomain, [vk, "not_a_kernel"])

    def test_sola_rejects_wrong_domain_measurement(self, unit_box):
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

    def test_sola_rejects_zero_measurements(self, space):
        """
        An empty measurement list must raise ValueError at construction time,
        guarding against an IndexError in _dual_mapping on empty coordinates.
        The guard fires before the len != codomain.dim check.
        """
        with pytest.raises(ValueError, match="at least one measurement"):
            SOLAOperator3D(space, EuclideanSpace(1), [])

    def test_sola_measurements_property_returns_copy(self, unit_box, space):
        """
        The .measurements property returns a copy; external mutation is safe.
        """
        vk = VolumeKernel3D(space, const_fn(unit_box, 1.0))
        codomain = EuclideanSpace(1)
        op = SOLAOperator3D(space, codomain, [vk])

        meas = op.measurements
        meas.clear()
        assert len(op.measurements) == 1


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

    The result is the weighted sum of the measurement forms.
    """

    def test_sola_dual_uses_codomain_dual_coordinates(self, unit_box, space):
        """
        For measurements m₁ (kernel=1), m₂ (kernel=x₀) and dual element
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
