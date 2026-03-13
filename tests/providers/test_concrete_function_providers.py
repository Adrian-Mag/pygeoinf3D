"""Tests for concrete FunctionProvider3D implementations (Phase 2).

Covers:
- GaussianProvider3D
- BumpProvider3D
- BoxcarProvider3D

Each family is tested for:
  - construction (valid and invalid inputs)
  - pointwise and batch evaluation
  - parameter validation
  - provider indexing (dim, get_function, out-of-range errors)
  - domain identity preserved in produced Function3D
  - approximate normalization (∫ f dV ≈ 1 when normalized=True)
  - compatibility with FunctionToKernelAdapter
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from pygeoinf3d.core.functions import Function3D
from pygeoinf3d.geometries.box.domain import BoxDomain
from pygeoinf3d.providers.base import (
    FunctionProvider3D,
    FunctionToKernelAdapter,
)
from pygeoinf3d.spaces.lebesgue import Lebesgue3D
from pygeoinf3d.spaces.forms import VolumeKernel3D


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def unit_box():
    """Unit cube [0,1]³."""
    return BoxDomain(np.zeros(3), np.ones(3))


@pytest.fixture
def wide_box():
    """Larger box [-2,2]³ for bump normalisation tests."""
    return BoxDomain(-2.0 * np.ones(3), 2.0 * np.ones(3))


@pytest.fixture
def lebesgue_unit(unit_box):
    return Lebesgue3D(unit_box)


@pytest.fixture
def centers_3():
    """Three centres well inside the unit cube."""
    return np.array([
        [0.25, 0.25, 0.25],
        [0.50, 0.50, 0.50],
        [0.75, 0.75, 0.75],
    ])


@pytest.fixture
def single_center():
    """Single centre as shape-(3,) array."""
    return np.array([0.5, 0.5, 0.5])


# ===========================================================================
# GaussianProvider3D
# ===========================================================================

class TestGaussianProvider3DImport:
    def test_importable(self):
        from pygeoinf3d.providers.function_providers import GaussianProvider3D
        assert GaussianProvider3D is not None

    def test_exported_in_all(self):
        import pygeoinf3d.providers.function_providers as fp
        assert "GaussianProvider3D" in fp.__all__


class TestGaussianProvider3DConstruction:
    def test_basic_construction_scalar_sigma(self, unit_box, centers_3):
        from pygeoinf3d.providers.function_providers import GaussianProvider3D
        p = GaussianProvider3D(unit_box, centers_3, sigmas=0.1)
        assert p.dim == 3

    def test_construction_sigma_per_axis(self, unit_box, centers_3):
        from pygeoinf3d.providers.function_providers import GaussianProvider3D
        p = GaussianProvider3D(
            unit_box,
            centers_3,
            sigmas=np.array([0.1, 0.2, 0.15]),
        )
        assert p.dim == 3

    def test_construction_sigma_per_kernel(self, unit_box, centers_3):
        from pygeoinf3d.providers.function_providers import GaussianProvider3D
        sigmas = np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [0.15, 0.1, 0.2]])
        p = GaussianProvider3D(unit_box, centers_3, sigmas=sigmas)
        assert p.dim == 3

    def test_single_center_shape_3(self, unit_box, single_center):
        from pygeoinf3d.providers.function_providers import GaussianProvider3D
        p = GaussianProvider3D(unit_box, single_center, sigmas=0.1)
        assert p.dim == 1

    def test_invalid_centers_shape_raises(self, unit_box):
        from pygeoinf3d.providers.function_providers import GaussianProvider3D
        with pytest.raises(ValueError, match="centers"):
            GaussianProvider3D(unit_box, np.ones((3, 2)), sigmas=0.1)

    def test_invalid_sigma_negative_raises(self, unit_box, centers_3):
        from pygeoinf3d.providers.function_providers import GaussianProvider3D
        with pytest.raises(ValueError, match="sigma"):
            GaussianProvider3D(unit_box, centers_3, sigmas=-0.1)

    def test_invalid_sigma_zero_raises(self, unit_box, centers_3):
        from pygeoinf3d.providers.function_providers import GaussianProvider3D
        with pytest.raises(ValueError, match="sigma"):
            GaussianProvider3D(unit_box, centers_3, sigmas=0.0)

    def test_invalid_sigma_shape_raises(self, unit_box, centers_3):
        from pygeoinf3d.providers.function_providers import GaussianProvider3D
        with pytest.raises(ValueError, match="sigma"):
            GaussianProvider3D(
                unit_box,
                centers_3,
                sigmas=np.array([0.1, 0.2]),
            )


class TestGaussianProvider3DIndexing:
    @pytest.fixture
    def provider(self, unit_box, centers_3):
        from pygeoinf3d.providers.function_providers import GaussianProvider3D
        return GaussianProvider3D(unit_box, centers_3, sigmas=0.1)

    def test_is_function_provider(self, provider):
        assert isinstance(provider, FunctionProvider3D)

    def test_get_function_returns_function3d(self, provider):
        f = provider.get_function(0)
        assert isinstance(f, Function3D)

    def test_get_function_all_indices(self, provider):
        for i in range(provider.dim):
            assert isinstance(provider.get_function(i), Function3D)

    def test_domain_identity_preserved(self, provider, unit_box):
        f = provider.get_function(1)
        assert f.domain is unit_box

    def test_out_of_range_raises_index_error(self, provider):
        with pytest.raises(IndexError):
            provider.get_function(provider.dim)

    def test_negative_index_raises_index_error(self, provider):
        with pytest.raises(IndexError):
            provider.get_function(-1)


class TestGaussianProvider3DEvaluation:
    @pytest.fixture
    def provider(self, unit_box, centers_3):
        from pygeoinf3d.providers.function_providers import GaussianProvider3D
        return GaussianProvider3D(unit_box, centers_3, sigmas=0.1)

    def test_pointwise_single_point(self, provider):
        f = provider.get_function(1)
        result = f(np.array([0.5, 0.5, 0.5]))
        assert np.isfinite(result)
        assert float(result) > 0

    def test_batch_evaluation(self, provider):
        f = provider.get_function(1)
        pts = np.array([
            [0.5, 0.5, 0.5],
            [0.3, 0.3, 0.3],
            [0.0, 0.0, 0.0],
        ])
        result = np.asarray(f(pts))
        assert result.shape == (3,)
        assert np.all(np.isfinite(result))
        # center is [0.5, 0.5, 0.5] so value there is max
        assert result[0] > result[1] > result[2]

    def test_maximum_at_center(self, provider):
        """Gaussian has maximum at its center."""
        f = provider.get_function(0)
        c = np.array([0.25, 0.25, 0.25])
        val_at_center = float(f(c))
        pts = np.random.default_rng(42).uniform(0, 1, (50, 3))
        vals = np.asarray(f(pts))
        assert val_at_center >= np.max(vals) - 1e-12

    def test_far_point_is_small(self, provider):
        """Far from center value should be tiny for sigma=0.1."""
        f = provider.get_function(0)
        far = np.array([0.9, 0.9, 0.9])
        assert float(f(far)) < 1e-10

    def test_non_normalized_positive(self, unit_box, centers_3):
        from pygeoinf3d.providers.function_providers import GaussianProvider3D
        p = GaussianProvider3D(
            unit_box,
            centers_3,
            sigmas=0.1,
            normalized=False,
        )
        f = p.get_function(1)
        val = float(f(np.array([0.5, 0.5, 0.5])))
        assert val == pytest.approx(1.0, abs=1e-12)

    def test_normalized_integral_approx_one(self, wide_box):
        """∫ gaussian dV ≈ 1 when center is well inside the domain.

        Uses BoxDomain.integrate_volume for a large box so the tails are
        negligible.
        """
        from pygeoinf3d.providers.function_providers import GaussianProvider3D
        center = np.array([[0.0, 0.0, 0.0]])
        p = GaussianProvider3D(wide_box, center, sigmas=0.3, normalized=True)
        f = p.get_function(0)
        integral = wide_box.integrate_volume(f, n=40)
        npt.assert_allclose(integral, 1.0, rtol=0.05, atol=0.0)


class TestGaussianProvider3DAdapterCompatibility:
    def test_adapter_produces_volume_kernel(
        self,
        unit_box,
        centers_3,
        lebesgue_unit,
    ):
        from pygeoinf3d.providers.function_providers import GaussianProvider3D
        p = GaussianProvider3D(unit_box, centers_3, sigmas=0.1)
        adapter = FunctionToKernelAdapter(p, lebesgue_unit)
        assert adapter.dim == 3
        k = adapter.get_kernel(0)
        assert isinstance(k, VolumeKernel3D)

    def test_kernel_evaluation_finite(
        self,
        unit_box,
        centers_3,
        lebesgue_unit,
    ):
        from pygeoinf3d.providers.function_providers import GaussianProvider3D
        p = GaussianProvider3D(unit_box, centers_3, sigmas=0.1)
        adapter = FunctionToKernelAdapter(p, lebesgue_unit)
        zero_fn = lebesgue_unit.zero
        # VolumeKernel3D(space, f)(g) ≈ ∫ f·g dV — with zero it should be 0
        k = adapter.get_kernel(1)
        val = k(zero_fn)
        assert val == pytest.approx(0.0, abs=1e-12)


# ===========================================================================
# BumpProvider3D
# ===========================================================================

class TestBumpProvider3DImport:
    def test_importable(self):
        from pygeoinf3d.providers.function_providers import BumpProvider3D
        assert BumpProvider3D is not None

    def test_exported_in_all(self):
        import pygeoinf3d.providers.function_providers as fp
        assert "BumpProvider3D" in fp.__all__


class TestBumpProvider3DConstruction:
    def test_basic_construction(self, unit_box, centers_3):
        from pygeoinf3d.providers.function_providers import BumpProvider3D
        p = BumpProvider3D(unit_box, centers_3, widths=0.2)
        assert p.dim == 3

    def test_single_center(self, unit_box, single_center):
        from pygeoinf3d.providers.function_providers import BumpProvider3D
        p = BumpProvider3D(unit_box, single_center, widths=0.2)
        assert p.dim == 1

    def test_per_axis_widths(self, unit_box, centers_3):
        from pygeoinf3d.providers.function_providers import BumpProvider3D
        p = BumpProvider3D(
            unit_box,
            centers_3,
            widths=np.array([0.2, 0.3, 0.15]),
        )
        assert p.dim == 3

    def test_per_kernel_widths(self, unit_box, centers_3):
        from pygeoinf3d.providers.function_providers import BumpProvider3D
        widths = np.array([[0.2, 0.2, 0.2], [0.3, 0.3, 0.3], [0.15, 0.2, 0.1]])
        p = BumpProvider3D(unit_box, centers_3, widths=widths)
        assert p.dim == 3

    def test_invalid_centers_raises(self, unit_box):
        from pygeoinf3d.providers.function_providers import BumpProvider3D
        with pytest.raises(ValueError, match="centers"):
            BumpProvider3D(unit_box, np.ones((3, 4)), widths=0.2)

    def test_invalid_widths_negative_raises(self, unit_box, centers_3):
        from pygeoinf3d.providers.function_providers import BumpProvider3D
        with pytest.raises(ValueError, match="width"):
            BumpProvider3D(unit_box, centers_3, widths=-0.1)

    def test_invalid_k_non_positive_raises(self, unit_box, centers_3):
        from pygeoinf3d.providers.function_providers import BumpProvider3D
        with pytest.raises(ValueError, match="k"):
            BumpProvider3D(unit_box, centers_3, widths=0.2, k=0.0)


class TestBumpProvider3DIndexing:
    @pytest.fixture
    def provider(self, unit_box, centers_3):
        from pygeoinf3d.providers.function_providers import BumpProvider3D
        return BumpProvider3D(unit_box, centers_3, widths=0.2)

    def test_is_function_provider(self, provider):
        assert isinstance(provider, FunctionProvider3D)

    def test_get_function_returns_function3d(self, provider):
        assert isinstance(provider.get_function(0), Function3D)

    def test_domain_identity_preserved(self, provider, unit_box):
        assert provider.get_function(0).domain is unit_box

    def test_out_of_range_raises(self, provider):
        with pytest.raises(IndexError):
            provider.get_function(provider.dim)

    def test_negative_raises(self, provider):
        with pytest.raises(IndexError):
            provider.get_function(-1)


class TestBumpProvider3DEvaluation:
    @pytest.fixture
    def provider(self, unit_box, centers_3):
        from pygeoinf3d.providers.function_providers import BumpProvider3D
        return BumpProvider3D(unit_box, centers_3, widths=0.3)

    def test_pointwise_at_center_positive(self, provider):
        f = provider.get_function(1)
        val = float(f(np.array([0.5, 0.5, 0.5])))
        assert val > 0

    def test_outside_support_is_zero(self, provider):
        """Points outside the compact support must be exactly 0."""
        f = provider.get_function(0)  # center=[0.25,0.25,0.25], width=0.3
        # half-width = 0.15, so x=0.7 is outside support
        val = float(f(np.array([0.7, 0.7, 0.7])))
        assert val == pytest.approx(0.0, abs=1e-15)

    def test_batch_inside_outside(self, provider):
        f = provider.get_function(0)  # center=[0.25,0.25,0.25], width=0.3
        pts = np.array([
            [0.25, 0.25, 0.25],   # inside, at center
            [0.39, 0.39, 0.39],   # still inside (half-width 0.15)
            [0.42, 0.42, 0.42],   # outside
        ])
        result = np.asarray(f(pts))
        assert result.shape == (3,)
        assert result[0] > 0
        assert result[2] == pytest.approx(0.0, abs=1e-15)

    def test_compact_support_exact_zero(self, unit_box):
        """The function must be exactly 0 beyond support — not just small."""
        from pygeoinf3d.providers.function_providers import BumpProvider3D
        center = np.array([[0.5, 0.5, 0.5]])
        p = BumpProvider3D(unit_box, center, widths=0.4)  # support [0.3,0.7]^3
        f = p.get_function(0)
        outside = np.array([
            [0.0, 0.5, 0.5],
            [1.0, 0.5, 0.5],
            [0.5, 0.5, 0.0],
        ])
        vals = np.asarray(f(outside))
        npt.assert_array_equal(vals, 0.0)

    def test_normalized_integral_approx_one(self, wide_box):
        """∫ bump dV ≈ 1 when center is well inside domain."""
        from pygeoinf3d.providers.function_providers import BumpProvider3D
        center = np.array([[0.0, 0.0, 0.0]])
        p = BumpProvider3D(wide_box, center, widths=1.0, normalized=True)
        f = p.get_function(0)
        integral = wide_box.integrate_volume(f, n=40)
        npt.assert_allclose(integral, 1.0, rtol=0.05, atol=0.0)

    def test_non_normalized_at_center(self, unit_box):
        """Non-normalized bump at center should be exp(0) product = 1."""
        from pygeoinf3d.providers.function_providers import BumpProvider3D
        center = np.array([[0.5, 0.5, 0.5]])
        p = BumpProvider3D(unit_box, center, widths=0.4, normalized=False)
        f = p.get_function(0)
        val = float(f(np.array([0.5, 0.5, 0.5])))
        # At center, t=0 for all axes, so exp(0) = 1, product = 1
        assert val == pytest.approx(1.0, abs=1e-12)


class TestBumpProvider3DAdapterCompatibility:
    def test_adapter_produces_kernel(self, unit_box, centers_3, lebesgue_unit):
        from pygeoinf3d.providers.function_providers import BumpProvider3D
        p = BumpProvider3D(unit_box, centers_3, widths=0.15)
        adapter = FunctionToKernelAdapter(p, lebesgue_unit)
        k = adapter.get_kernel(0)
        assert isinstance(k, VolumeKernel3D)


# ===========================================================================
# BoxcarProvider3D
# ===========================================================================

class TestBoxcarProvider3DImport:
    def test_importable(self):
        from pygeoinf3d.providers.function_providers import BoxcarProvider3D
        assert BoxcarProvider3D is not None

    def test_exported_in_all(self):
        import pygeoinf3d.providers.function_providers as fp
        assert "BoxcarProvider3D" in fp.__all__


class TestBoxcarProvider3DConstruction:
    def test_basic_construction(self, unit_box, centers_3):
        from pygeoinf3d.providers.function_providers import BoxcarProvider3D
        p = BoxcarProvider3D(unit_box, centers_3, widths=0.2)
        assert p.dim == 3

    def test_single_center(self, unit_box, single_center):
        from pygeoinf3d.providers.function_providers import BoxcarProvider3D
        p = BoxcarProvider3D(unit_box, single_center, widths=0.3)
        assert p.dim == 1

    def test_per_axis_widths(self, unit_box, centers_3):
        from pygeoinf3d.providers.function_providers import BoxcarProvider3D
        p = BoxcarProvider3D(
            unit_box,
            centers_3,
            widths=np.array([0.2, 0.3, 0.4]),
        )
        assert p.dim == 3

    def test_invalid_centers_raises(self, unit_box):
        from pygeoinf3d.providers.function_providers import BoxcarProvider3D
        with pytest.raises(ValueError, match="centers"):
            BoxcarProvider3D(unit_box, np.ones((3, 2)), widths=0.2)

    def test_invalid_widths_zero_raises(self, unit_box, centers_3):
        from pygeoinf3d.providers.function_providers import BoxcarProvider3D
        with pytest.raises(ValueError, match="width"):
            BoxcarProvider3D(unit_box, centers_3, widths=0.0)

    def test_invalid_widths_negative_raises(self, unit_box, centers_3):
        from pygeoinf3d.providers.function_providers import BoxcarProvider3D
        with pytest.raises(ValueError, match="width"):
            BoxcarProvider3D(unit_box, centers_3, widths=-0.1)

    def test_invalid_widths_shape_raises(self, unit_box, centers_3):
        from pygeoinf3d.providers.function_providers import BoxcarProvider3D
        with pytest.raises(ValueError, match="width"):
            BoxcarProvider3D(unit_box, centers_3, widths=np.array([0.2, 0.3]))

    def test_per_kernel_widths_n3(self, unit_box, centers_3):
        """widths shaped (n, 3) gives each kernel its own per-axis widths."""
        from pygeoinf3d.providers.function_providers import BoxcarProvider3D
        widths = np.array([
            [0.2, 0.3, 0.4],
            [0.1, 0.1, 0.2],
            [0.4, 0.2, 0.3],
        ])
        p = BoxcarProvider3D(unit_box, centers_3, widths=widths)
        assert p.dim == 3
        # Each kernel's function should evaluate to the expected height when
        # normalized=True: height = 1 / (w0 * w1 * w2)
        for i in range(3):
            expected_h = 1.0 / (widths[i, 0] * widths[i, 1] * widths[i, 2])
            f = p.get_function(i)
            val = float(f(centers_3[i]))
            assert val == pytest.approx(expected_h, rel=1e-12)


class TestBoxcarProvider3DIndexing:
    @pytest.fixture
    def provider(self, unit_box, centers_3):
        from pygeoinf3d.providers.function_providers import BoxcarProvider3D
        return BoxcarProvider3D(unit_box, centers_3, widths=0.2)

    def test_is_function_provider(self, provider):
        assert isinstance(provider, FunctionProvider3D)

    def test_get_function_returns_function3d(self, provider):
        assert isinstance(provider.get_function(0), Function3D)

    def test_all_indices(self, provider):
        for i in range(provider.dim):
            assert isinstance(provider.get_function(i), Function3D)

    def test_domain_identity_preserved(self, provider, unit_box):
        assert provider.get_function(2).domain is unit_box

    def test_out_of_range_raises(self, provider):
        with pytest.raises(IndexError):
            provider.get_function(provider.dim)

    def test_negative_raises(self, provider):
        with pytest.raises(IndexError):
            provider.get_function(-1)


class TestBoxcarProvider3DEvaluation:
    def test_inside_support_is_positive(self, unit_box, centers_3):
        from pygeoinf3d.providers.function_providers import BoxcarProvider3D
        p = BoxcarProvider3D(unit_box, centers_3, widths=0.3)
        f = p.get_function(1)  # center=[0.5,0.5,0.5], half-width=0.15
        val = float(f(np.array([0.5, 0.5, 0.5])))
        assert val > 0

    def test_outside_support_is_zero(self, unit_box, centers_3):
        from pygeoinf3d.providers.function_providers import BoxcarProvider3D
        p = BoxcarProvider3D(unit_box, centers_3, widths=0.2)
        f = p.get_function(0)  # center=[0.25,0.25,0.25], half-width=0.1
        val = float(f(np.array([0.5, 0.5, 0.5])))
        assert val == pytest.approx(0.0, abs=1e-15)

    def test_constant_inside_support(self, unit_box):
        """Boxcar must have constant value inside its support."""
        from pygeoinf3d.providers.function_providers import BoxcarProvider3D
        center = np.array([[0.5, 0.5, 0.5]])
        p = BoxcarProvider3D(unit_box, center, widths=0.4, normalized=False)
        f = p.get_function(0)
        pts = np.array([
            [0.5, 0.5, 0.5],
            [0.4, 0.6, 0.5],
            [0.5, 0.4, 0.7],
        ])
        vals = np.asarray(f(pts))
        npt.assert_allclose(vals, vals[0], rtol=1e-14)

    def test_batch_evaluation(self, unit_box, centers_3):
        from pygeoinf3d.providers.function_providers import BoxcarProvider3D
        p = BoxcarProvider3D(unit_box, centers_3, widths=0.2)
        f = p.get_function(1)
        pts = np.array([
            [0.5, 0.5, 0.5],   # inside
            [0.0, 0.0, 0.0],   # outside
        ])
        result = np.asarray(f(pts))
        assert result.shape == (2,)
        assert result[0] > 0
        assert result[1] == pytest.approx(0.0, abs=1e-15)

    def test_normalized_value(self, unit_box):
        """Normalized boxcar: value inside = 1 / volume."""
        from pygeoinf3d.providers.function_providers import BoxcarProvider3D
        center = np.array([[0.5, 0.5, 0.5]])
        widths = np.array([0.4, 0.2, 0.6])
        p = BoxcarProvider3D(unit_box, center, widths=widths, normalized=True)
        f = p.get_function(0)
        expected_h = 1.0 / (0.4 * 0.2 * 0.6)
        val = float(f(np.array([0.5, 0.5, 0.5])))
        assert val == pytest.approx(expected_h, rel=1e-12)

    def test_normalized_integral_approx_one(self, unit_box):
        """∫ boxcar dV ≈ 1 when box is fully inside domain."""
        from pygeoinf3d.providers.function_providers import BoxcarProvider3D
        center = np.array([[0.5, 0.5, 0.5]])
        p = BoxcarProvider3D(unit_box, center, widths=0.4, normalized=True)
        f = p.get_function(0)
        integral = unit_box.integrate_volume(f, n=40)
        npt.assert_allclose(integral, 1.0, rtol=0.05, atol=0.0)

    def test_non_normalized_default_height_one(self, unit_box):
        """Non-normalized boxcar has height 1.0 inside."""
        from pygeoinf3d.providers.function_providers import BoxcarProvider3D
        center = np.array([[0.5, 0.5, 0.5]])
        p = BoxcarProvider3D(unit_box, center, widths=0.4, normalized=False)
        f = p.get_function(0)
        val = float(f(np.array([0.5, 0.5, 0.5])))
        assert val == pytest.approx(1.0, abs=1e-12)


class TestBoxcarProvider3DAdapterCompatibility:
    def test_adapter_produces_kernel(self, unit_box, centers_3, lebesgue_unit):
        from pygeoinf3d.providers.function_providers import BoxcarProvider3D
        p = BoxcarProvider3D(unit_box, centers_3, widths=0.2)
        adapter = FunctionToKernelAdapter(p, lebesgue_unit)
        k = adapter.get_kernel(0)
        assert isinstance(k, VolumeKernel3D)

    def test_kernel_callable_on_boxcar_function(self, unit_box, lebesgue_unit):
        """The kernel of a boxcar is callable as a VolumeKernel3D."""
        from pygeoinf3d.providers.function_providers import BoxcarProvider3D
        center = np.array([[0.5, 0.5, 0.5]])
        p = BoxcarProvider3D(unit_box, center, widths=0.2, normalized=True)
        adapter = FunctionToKernelAdapter(p, lebesgue_unit)
        k = adapter.get_kernel(0)
        # Evaluate against the zero function; confirms call doesn't throw
        one_fn = lebesgue_unit.zero
        assert k(one_fn) == pytest.approx(0.0, abs=1e-12)
