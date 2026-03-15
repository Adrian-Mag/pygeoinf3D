import matplotlib.pyplot as plt
import numpy as np
import pytest
from typing import Any, cast

import pygeoinf3d.plot as plot_module

from pygeoinf3d.core.functions import Function3D
from pygeoinf3d.geometries.box.domain import BoxDomain
from pygeoinf3d.plot import (
    evaluate_function3d_on_axis_slice,
    evaluate_function3d_on_regular_grid,
    plot_function3d_axis_slice,
    plot_function3d_axis_slider,
    plot_function3d_volume_dropdown,
    plot_function3d_isosurface,
    compare_function3d_axis_sliders,
)


@pytest.fixture
def box_domain():
    return BoxDomain(np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0]))


@pytest.fixture
def linear_function(box_domain):
    def f(x):
        x = np.asarray(x)
        if x.ndim == 1:
            return x[0] + x[1] + x[2]
        return x[:, 0] + x[:, 1] + x[:, 2]
    return Function3D(box_domain, evaluate_callable=f)


def test_evaluate_slice_z_center(linear_function):
    # Slice at z=0.5; resolution=10 gives a 10×10 grid over x and y
    grid_u, grid_v, values = evaluate_function3d_on_axis_slice(
        linear_function, axis='z', coord=0.5, resolution=10
    )
    assert grid_u.shape == (10, 10)
    assert grid_v.shape == (10, 10)
    assert values.shape == (10, 10)

    # Check bounds
    np.testing.assert_allclose(grid_u.min(), 0.0)
    np.testing.assert_allclose(grid_u.max(), 1.0)
    np.testing.assert_allclose(grid_v.min(), 0.0)
    np.testing.assert_allclose(grid_v.max(), 1.0)

    # Corner (0, 0) -> x=0, y=0, z=0.5 => value = 0.5
    u0 = grid_u[0, 0]
    v0 = grid_v[0, 0]
    val0 = values[0, 0]
    np.testing.assert_allclose(val0, u0 + v0 + 0.5)


def test_evaluate_slice_x_axis(linear_function):
    # Slice at x=0; returned grids are y (u) and z (v)
    grid_u, grid_v, values = evaluate_function3d_on_axis_slice(
        linear_function, axis=0, coord=0.0, resolution=(5, 5)
    )
    assert values.shape == (5, 5)
    np.testing.assert_allclose(values, grid_u + grid_v)


def test_plot_slice_smoke(linear_function):
    plt.switch_backend('Agg')

    fig, ax, payload = plot_function3d_axis_slice(
        linear_function, axis='y', coord=0.2, show_plot=False
    )
    assert fig is not None
    assert ax is not None
    assert isinstance(payload, dict)
    assert 'grid_u' in payload
    assert 'grid_v' in payload
    assert 'values' in payload
    assert 'axis_idx' in payload
    assert payload['axis_idx'] == 1  # y-axis

    assert ax.get_xlabel() != ""
    assert ax.get_ylabel() != ""
    plt.close(fig)


def test_invalid_input(linear_function):
    with pytest.raises(ValueError, match="Invalid axis"):
        evaluate_function3d_on_axis_slice(linear_function, axis='w')

    with pytest.raises(ValueError, match="outside domain bounds"):
        evaluate_function3d_on_axis_slice(linear_function, axis='x', coord=2.0)


def test_resolution_validation(linear_function):
    with pytest.raises(ValueError, match="positive int"):
        evaluate_function3d_on_axis_slice(
            linear_function, axis='z', resolution=0
        )

    with pytest.raises(ValueError, match="positive int"):
        evaluate_function3d_on_axis_slice(
            linear_function, axis='z', resolution=-5
        )

    with pytest.raises(ValueError, match="exactly 2 elements"):
        evaluate_function3d_on_axis_slice(
            linear_function,
            axis='z',
            resolution=(5, 5, 5),  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="positive int"):
        evaluate_function3d_on_axis_slice(
            linear_function, axis='z', resolution=(0, 5)
        )


def test_default_coord(linear_function):
    # Default coord is domain center; z=0.5, values = x + y + 0.5
    grid_u, grid_v, values = evaluate_function3d_on_axis_slice(
        linear_function, axis='z', resolution=3
    )
    mid = 1  # center index in a 3-point grid
    np.testing.assert_allclose(grid_u[mid, mid], 0.5)
    np.testing.assert_allclose(grid_v[mid, mid], 0.5)
    np.testing.assert_allclose(values[mid, mid], 1.5)


def test_evaluate_regular_grid(linear_function):
    X, Y, Z, values = evaluate_function3d_on_regular_grid(
        linear_function,
        resolution=4,
    )

    assert X.shape == (4, 4, 4)
    assert Y.shape == (4, 4, 4)
    assert Z.shape == (4, 4, 4)
    assert values.shape == (4, 4, 4)
    np.testing.assert_allclose(values, X + Y + Z)


def test_evaluate_regular_grid_validation(linear_function):
    with pytest.raises(ValueError, match="integer >= 2"):
        evaluate_function3d_on_regular_grid(linear_function, resolution=1)


def test_plot_axis_slider_smoke(linear_function):
    go = pytest.importorskip("plotly.graph_objects")

    fig = cast(Any, plot_function3d_axis_slider(
        linear_function,
        axis='z',
        n_slices=5,
        resolution=6,
        show_plot=False,
    ))

    assert isinstance(fig, go.Figure)
    assert len(fig.frames) == 5
    assert fig.layout.sliders[0].active == 2
    assert fig.data[0].type == 'heatmap'
    assert fig.layout.xaxis.title.text == 'x coordinate'
    assert fig.layout.yaxis.title.text == 'y coordinate'


def test_plot_axis_slider_x_axis_labels(linear_function):
    pytest.importorskip("plotly.graph_objects")

    fig = cast(Any, plot_function3d_axis_slider(
        linear_function,
        axis='x',
        n_slices=4,
        resolution=(4, 6),
        show_plot=False,
    ))

    assert len(fig.frames) == 4
    assert fig.layout.xaxis.title.text == 'y coordinate'
    assert fig.layout.yaxis.title.text == 'z coordinate'
    assert fig.layout.sliders[0].currentvalue.prefix == 'x = '


def test_plot_axis_slider_y_axis_labels(linear_function):
    pytest.importorskip("plotly.graph_objects")

    fig = cast(Any, plot_function3d_axis_slider(
        linear_function,
        axis='y',
        n_slices=3,
        resolution=5,
        show_plot=False,
    ))

    assert len(fig.frames) == 3
    assert fig.layout.xaxis.title.text == 'x coordinate'
    assert fig.layout.yaxis.title.text == 'z coordinate'
    assert fig.layout.sliders[0].currentvalue.prefix == 'y = '


def test_plot_axis_slider_numpy_integer_axis(linear_function):
    pytest.importorskip("plotly.graph_objects")

    fig = cast(Any, plot_function3d_axis_slider(
        linear_function,
        axis=cast(Any, np.int64(2)),
        n_slices=2,
        resolution=4,
        show_plot=False,
    ))

    assert len(fig.frames) == 2
    assert fig.layout.sliders[0].currentvalue.prefix == 'z = '


def test_plot_axis_slider_validation(linear_function):
    with pytest.raises(
        ValueError,
        match="n_slices must be a positive integer",
    ):
        plot_function3d_axis_slider(
            linear_function,
            axis='z',
            n_slices=0,
            show_plot=False,
        )


def test_plot_axis_slider_requires_plotly(linear_function, monkeypatch):
    def _raise_import_error():
        raise ImportError("plotly missing")

    monkeypatch.setattr(
        plot_module,
        "_import_plotly_graph_objects",
        _raise_import_error,
    )

    with pytest.raises(ImportError, match="plotly missing"):
        plot_module.plot_function3d_axis_slider(
            linear_function,
            axis='z',
            show_plot=False,
        )


def test_plot_function3d_volume_dropdown_smoke(two_functions):
    go = pytest.importorskip("plotly.graph_objects")

    fig = cast(Any, plot_function3d_volume_dropdown(
        two_functions,
        resolution=5,
        show_plot=False,
        labels=["First", "Second"],
    ))

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2
    assert fig.data[0].type == 'volume'
    assert fig.data[0].visible is True
    assert fig.data[1].visible is False
    assert fig.layout.updatemenus[0].buttons[0].label == 'First'
    assert fig.layout.scene.xaxis.title.text == 'x'
    assert fig.layout.scene.yaxis.title.text == 'y'
    assert fig.layout.scene.zaxis.title.text == 'z'


def test_plot_function3d_volume_dropdown_validation(two_functions):
    with pytest.raises(ValueError, match="non-empty"):
        plot_function3d_volume_dropdown([], show_plot=False)

    with pytest.raises(ValueError, match="same length"):
        plot_function3d_volume_dropdown(
            two_functions,
            labels=['only one'],
            show_plot=False,
        )

    with pytest.raises(ValueError, match=r"interval \(0, 100\]"):
        plot_function3d_volume_dropdown(
            two_functions,
            robust_percentile=0.0,
            show_plot=False,
        )


def test_plot_function3d_volume_dropdown_requires_shared_bbox(box_domain):
    pytest.importorskip("plotly.graph_objects")

    other_domain = BoxDomain(
        np.array([0.0, 0.0, 0.0]),
        np.array([2.0, 2.0, 2.0]),
    )

    def f(x):
        x = np.asarray(x)
        return x[0] if x.ndim == 1 else x[:, 0]

    fn1 = Function3D(box_domain, evaluate_callable=f)
    fn2 = Function3D(other_domain, evaluate_callable=f)

    with pytest.raises(ValueError, match="bounding box"):
        plot_function3d_volume_dropdown(
            [fn1, fn2],
            show_plot=False,
        )


def test_plot_function3d_isosurface_smoke(linear_function):
    go = pytest.importorskip("plotly.graph_objects")

    fig = cast(Any, plot_function3d_isosurface(
        linear_function,
        resolution=5,
        n_levels=4,
        show_plot=False,
    ))

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].type == 'isosurface'


# ---------------------------------------------------------------------------
# Tests for compare_function3d_axis_sliders
# ---------------------------------------------------------------------------

@pytest.fixture
def two_functions(box_domain):
    """Two simple Function3D objects on the same unit-cube domain."""
    def f1(x):
        x = np.asarray(x)
        if x.ndim == 1:
            return x[0]
        return x[:, 0]

    def f2(x):
        x = np.asarray(x)
        if x.ndim == 1:
            return x[1]
        return x[:, 1]

    return [
        Function3D(box_domain, evaluate_callable=f1, name="f1"),
        Function3D(box_domain, evaluate_callable=f2, name="f2"),
    ]


def test_compare_function3d_axis_sliders_smoke(two_functions):
    go = pytest.importorskip("plotly.graph_objects")

    fig = cast(Any, compare_function3d_axis_sliders(
        two_functions,
        axis='z',
        n_slices=4,
        resolution=5,
        show_plot=False,
    ))

    assert isinstance(fig, go.Figure)
    # Each panel gives one trace in the initial figure data
    assert len(fig.data) == 2
    # Each frame updates both traces
    assert len(fig.frames) == 4
    for frame in fig.frames:
        assert len(frame.data) == 2
        assert list(frame.traces) == [0, 1]
    # Slider exists and has correct prefix
    assert fig.layout.sliders[0].currentvalue.prefix == 'z = '


def test_compare_function3d_axis_sliders_titles(two_functions):
    pytest.importorskip("plotly.graph_objects")

    custom_titles = ["Panel A", "Panel B"]
    fig = cast(Any, compare_function3d_axis_sliders(
        two_functions,
        axis='z',
        n_slices=3,
        resolution=4,
        show_plot=False,
        titles=custom_titles,
    ))

    annotation_texts = [a.text for a in fig.layout.annotations]
    for t in custom_titles:
        assert t in annotation_texts


def test_compare_function3d_axis_sliders_single_panel(linear_function):
    go = pytest.importorskip("plotly.graph_objects")

    fig = cast(Any, compare_function3d_axis_sliders(
        [linear_function],
        axis='x',
        n_slices=3,
        resolution=4,
        show_plot=False,
    ))

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert len(fig.frames) == 3
    assert fig.layout.sliders[0].currentvalue.prefix == 'x = '


def test_compare_function3d_axis_sliders_axis_labels(two_functions):
    pytest.importorskip("plotly.graph_objects")

    fig = cast(Any, compare_function3d_axis_sliders(
        two_functions,
        axis='z',
        n_slices=2,
        resolution=3,
        show_plot=False,
    ))

    # For z-axis, the slice plane uses x (u) and y (v)
    assert fig.layout.xaxis.title.text == 'x coordinate'
    assert fig.layout.yaxis.title.text == 'y coordinate'


def test_compare_function3d_axis_sliders_per_panel_color_limits(box_domain):
    """Each panel keeps its own colour scale when magnitudes differ."""
    pytest.importorskip("plotly.graph_objects")

    # f1 ranges over [0, 1], f2 over [0, 100] — very different magnitudes
    def f1(x):
        x = np.asarray(x)
        return x[0] if x.ndim == 1 else x[:, 0]

    def f2(x):
        x = np.asarray(x)
        return 100.0 * (x[0] if x.ndim == 1 else x[:, 0])

    from pygeoinf3d.core.functions import Function3D
    fns = [
        Function3D(box_domain, evaluate_callable=f1),
        Function3D(box_domain, evaluate_callable=f2),
    ]

    fig = cast(Any, compare_function3d_axis_sliders(
        fns,
        axis='z',
        n_slices=3,
        resolution=4,
        show_plot=False,
    ))

    # Per-panel scaling: the two panels must have different zmax values
    zmaxs = [trace.zmax for trace in fig.data]
    assert zmaxs[1] > 10 * zmaxs[0], (
        "Panel 2 (f2) should have a much larger zmax than panel 1 (f1)"
    )


def test_compare_function3d_axis_sliders_global_color_limits_via_vmin_vmax(
    two_functions,
):
    """Passing explicit vmin/vmax locks all panels to the same colour scale."""
    pytest.importorskip("plotly.graph_objects")

    fig = cast(Any, compare_function3d_axis_sliders(
        two_functions,
        axis='z',
        n_slices=3,
        resolution=4,
        show_plot=False,
        vmin=-1.0,
        vmax=1.0,
    ))

    for trace in fig.data:
        assert trace.zmin == -1.0
        assert trace.zmax == 1.0


def test_compare_function3d_axis_sliders_validation_empty(box_domain):
    with pytest.raises(ValueError, match="non-empty"):
        compare_function3d_axis_sliders([], show_plot=False)


def test_compare_function3d_axis_sliders_validation_titles_length(
    two_functions,
):
    pytest.importorskip("plotly.graph_objects")

    with pytest.raises(ValueError, match="same length"):
        compare_function3d_axis_sliders(
            two_functions,
            titles=["Only one title"],
            n_slices=2,
            resolution=3,
            show_plot=False,
        )


def test_compare_function3d_axis_sliders_validation_bbox_mismatch(box_domain):
    pytest.importorskip("plotly.graph_objects")

    other_domain = BoxDomain(
        np.array([0.0, 0.0, 0.0]),
        np.array([2.0, 2.0, 2.0]),
    )

    def f(x):
        x = np.asarray(x)
        return x[0] if x.ndim == 1 else x[:, 0]

    fn1 = Function3D(box_domain, evaluate_callable=f)
    fn2 = Function3D(other_domain, evaluate_callable=f)

    with pytest.raises(ValueError, match="bounding box"):
        compare_function3d_axis_sliders(
            [fn1, fn2],
            n_slices=2,
            resolution=3,
            show_plot=False,
        )


def test_compare_function3d_axis_sliders_requires_plotly(
    two_functions,
    monkeypatch,
):
    def _raise_import_error():
        raise ImportError("plotly missing")

    monkeypatch.setattr(
        plot_module,
        "_import_plotly_graph_objects",
        _raise_import_error,
    )

    with pytest.raises(ImportError, match="plotly missing"):
        plot_module.compare_function3d_axis_sliders(
            two_functions,
            show_plot=False,
        )


# ---------------------------------------------------------------------------
# Phase 2: KL fast-path integration tests
# ---------------------------------------------------------------------------


@pytest.fixture
def kl_sampler(box_domain):
    """KLSampler3D with tensor-product structure for fast-path testing."""
    from pygeoinf3d.spaces.lebesgue import Lebesgue3D
    from pygeoinf3d.providers.tensor_product import TensorProductProvider
    from pygeoinf3d.operators.inverse_laplacian_box import InverseBoxLaplacian
    from pygeoinf3d.sampling import KLSampler3D

    space = Lebesgue3D(box_domain)
    provider = TensorProductProvider(
        box_domain, truncation=(3, 3, 3), bc_type='dirichlet'
    )
    inv_lap = InverseBoxLaplacian(space, provider)
    return KLSampler3D(inv_lap, n_modes=5, rng=np.random.default_rng(42))


def test_kl_sample_has_fast_path_attributes(kl_sampler):
    """KL samples from a tensor-product operator carry fast-path callables."""
    sample = kl_sampler.sample()
    assert hasattr(sample, '_kl_fast_grid_evaluator'), (
        "KL sample should have _kl_fast_grid_evaluator"
    )
    assert hasattr(sample, '_kl_fast_slice_evaluator'), (
        "KL sample should have _kl_fast_slice_evaluator"
    )
    assert callable(sample._kl_fast_grid_evaluator)
    assert callable(sample._kl_fast_slice_evaluator)


def test_generic_function3d_has_no_fast_path(linear_function):
    """Generic Function3D objects do not carry fast-path callables."""
    assert not hasattr(linear_function, '_kl_fast_grid_evaluator')
    assert not hasattr(linear_function, '_kl_fast_slice_evaluator')


def test_kl_sample_from_xis_has_fast_path_attributes(kl_sampler):
    """sample_from_xis() also attaches fast-path callables."""
    xis = np.ones(kl_sampler.n_modes)
    sample = kl_sampler.sample_from_xis(xis)
    assert hasattr(sample, '_kl_fast_grid_evaluator')
    assert hasattr(sample, '_kl_fast_slice_evaluator')


def test_kl_fast_grid_numerically_matches_generic_path(kl_sampler):
    """Fast-path dispatch produces values numerically equivalent to generic path."""
    xis = np.random.default_rng(99).standard_normal(kl_sampler.n_modes)
    sample = kl_sampler.sample_from_xis(xis)

    # Fast path (dispatched automatically because sample has _kl_fast_grid_evaluator)
    _, _, _, vals_fast = evaluate_function3d_on_regular_grid(sample, resolution=5)

    # Generic path: wrap in a new Function3D without the fast-path attribute
    fn_generic = Function3D(
        sample.domain, evaluate_callable=lambda x: sample(x)
    )
    _, _, _, vals_gen = evaluate_function3d_on_regular_grid(fn_generic, resolution=5)

    np.testing.assert_allclose(vals_fast, vals_gen, rtol=1e-10, atol=1e-12)


def test_kl_fast_slice_numerically_matches_generic_path(kl_sampler):
    """Fast-path slice dispatch produces values equivalent to generic path."""
    xis = np.random.default_rng(77).standard_normal(kl_sampler.n_modes)
    sample = kl_sampler.sample_from_xis(xis)

    gu_fast, gv_fast, vals_fast = evaluate_function3d_on_axis_slice(
        sample, axis='z', coord=0.5, resolution=6
    )

    fn_generic = Function3D(
        sample.domain, evaluate_callable=lambda x: sample(x)
    )
    gu_gen, gv_gen, vals_gen = evaluate_function3d_on_axis_slice(
        fn_generic, axis='z', coord=0.5, resolution=6
    )

    np.testing.assert_allclose(vals_fast, vals_gen, rtol=1e-10, atol=1e-12)


def test_volume_dropdown_uses_kl_fast_path(kl_sampler):
    """KL samples use the fast-path evaluator in volume/dropdown plots."""
    pytest.importorskip("plotly.graph_objects")
    sample = kl_sampler.sample()

    fast_path_calls: list = []
    original_fast_path = sample._kl_fast_grid_evaluator

    def tracked_fast_path(resolution, plot_n_modes=None):
        fast_path_calls.append(resolution)
        return original_fast_path(resolution, plot_n_modes)

    sample._kl_fast_grid_evaluator = tracked_fast_path
    plot_function3d_volume_dropdown([sample], resolution=5, show_plot=False)

    assert len(fast_path_calls) == 1, (
        "Fast-path grid evaluator should be called exactly once"
    )
    assert fast_path_calls[0] == 5


def test_isosurface_uses_kl_fast_path(kl_sampler):
    """KL samples use the fast-path evaluator in isosurface plots."""
    pytest.importorskip("plotly.graph_objects")
    sample = kl_sampler.sample()

    fast_path_calls: list = []
    original_fast_path = sample._kl_fast_grid_evaluator

    def tracked_fast_path(resolution, plot_n_modes=None):
        fast_path_calls.append(resolution)
        return original_fast_path(resolution, plot_n_modes)

    sample._kl_fast_grid_evaluator = tracked_fast_path
    plot_function3d_isosurface(sample, resolution=5, show_plot=False)

    assert len(fast_path_calls) == 1, (
        "Fast-path grid evaluator should be called exactly once"
    )
    assert fast_path_calls[0] == 5


def test_axis_slider_uses_kl_fast_path(kl_sampler):
    """KL samples use the fast-path evaluator in axis-slider plots."""
    pytest.importorskip("plotly.graph_objects")
    sample = kl_sampler.sample()

    fast_path_calls: list = []
    original_fast_path = sample._kl_fast_slice_evaluator

    def tracked_fast_path(axis, coord, resolution, plot_n_modes=None):
        fast_path_calls.append((axis, coord))
        return original_fast_path(axis, coord, resolution, plot_n_modes)

    sample._kl_fast_slice_evaluator = tracked_fast_path
    plot_function3d_axis_slider(
        sample, axis='z', n_slices=3, resolution=5, show_plot=False
    )

    assert len(fast_path_calls) == 3, (
        "Fast-path slice evaluator should be called once per slice"
    )
    # All slices fix axis 2 (z)
    assert all(ax == 2 for ax, _ in fast_path_calls)


# ---------------------------------------------------------------------------
# Phase 3: plot_n_modes visualization knob
# ---------------------------------------------------------------------------


def test_plot_n_modes_validation(kl_sampler, linear_function):
    """plot_n_modes=0 or negative raises ValueError; None and valid ints are OK.

    Generic Function3D inputs silently ignore a valid plot_n_modes value
    rather than raising an error.
    """
    sample = kl_sampler.sample()

    with pytest.raises(ValueError, match="plot_n_modes"):
        evaluate_function3d_on_regular_grid(sample, resolution=4, plot_n_modes=0)

    with pytest.raises(ValueError, match="plot_n_modes"):
        evaluate_function3d_on_axis_slice(
            sample, axis='z', coord=0.5, resolution=4, plot_n_modes=-1
        )

    # None is always valid (no error)
    evaluate_function3d_on_regular_grid(sample, resolution=4, plot_n_modes=None)

    # Generic Function3D — valid plot_n_modes is silently ignored, no error,
    # and results are identical to omitting the argument
    _, _, _, vals_default = evaluate_function3d_on_regular_grid(
        linear_function, resolution=4
    )
    _, _, _, vals_with_pnm = evaluate_function3d_on_regular_grid(
        linear_function, resolution=4, plot_n_modes=2
    )
    np.testing.assert_allclose(vals_with_pnm, vals_default)

    _, _, vals_s_default = evaluate_function3d_on_axis_slice(
        linear_function, axis='z', coord=0.5, resolution=4
    )
    _, _, vals_s_pnm = evaluate_function3d_on_axis_slice(
        linear_function, axis='z', coord=0.5, resolution=4, plot_n_modes=2
    )
    np.testing.assert_allclose(vals_s_pnm, vals_s_default)


def test_plot_n_modes_changes_kl_sampling_budget_only_for_plotting(kl_sampler):
    """plot_n_modes truncates the KL sum used during visualization only.

    The underlying KLSampler3D.n_modes is unaffected; the Function3D sample
    itself still carries all modes for generic (non-fast-path) evaluation.
    """
    n_full = kl_sampler.n_modes  # 5 in the fixture
    xis = np.random.default_rng(321).standard_normal(n_full)
    sample = kl_sampler.sample_from_xis(xis)

    # Regular grid: plot_n_modes=1 should give different values than full
    _, _, _, vals_full = evaluate_function3d_on_regular_grid(sample, resolution=5)
    _, _, _, vals_trunc = evaluate_function3d_on_regular_grid(
        sample, resolution=5, plot_n_modes=1
    )
    assert not np.allclose(vals_full, vals_trunc), (
        "1-mode truncation should produce different values than full evaluation"
    )

    # Axis slice: same assertion
    _, _, vals_s_full = evaluate_function3d_on_axis_slice(
        sample, axis='z', coord=0.5, resolution=5
    )
    _, _, vals_s_trunc = evaluate_function3d_on_axis_slice(
        sample, axis='z', coord=0.5, resolution=5, plot_n_modes=1
    )
    assert not np.allclose(vals_s_full, vals_s_trunc), (
        "1-mode axis-slice truncation should differ from full evaluation"
    )

    # The sampler's n_modes is unchanged by visualization calls
    assert kl_sampler.n_modes == n_full

    # The underlying Function3D still evaluates with all modes generically
    fn_generic = Function3D(sample.domain, evaluate_callable=lambda x: sample(x))
    _, _, _, vals_generic = evaluate_function3d_on_regular_grid(fn_generic, resolution=5)
    np.testing.assert_allclose(vals_full, vals_generic, rtol=1e-10, atol=1e-12)


def test_plot_n_modes_full_equals_default(kl_sampler):
    """Setting plot_n_modes=n_modes gives results identical to the default."""
    n_full = kl_sampler.n_modes
    xis = np.random.default_rng(789).standard_normal(n_full)
    sample = kl_sampler.sample_from_xis(xis)

    _, _, _, vals_default = evaluate_function3d_on_regular_grid(sample, resolution=4)
    _, _, _, vals_capped = evaluate_function3d_on_regular_grid(
        sample, resolution=4, plot_n_modes=n_full
    )
    np.testing.assert_allclose(vals_capped, vals_default, rtol=1e-12, atol=1e-14)

    _, _, vals_s_default = evaluate_function3d_on_axis_slice(
        sample, axis='z', coord=0.5, resolution=4
    )
    _, _, vals_s_capped = evaluate_function3d_on_axis_slice(
        sample, axis='z', coord=0.5, resolution=4, plot_n_modes=n_full
    )
    np.testing.assert_allclose(vals_s_capped, vals_s_default, rtol=1e-12, atol=1e-14)


def test_plot_n_modes_threaded_through_volume_dropdown(kl_sampler):
    """plot_n_modes flows from plot_function3d_volume_dropdown to the fast path."""
    pytest.importorskip("plotly.graph_objects")
    sample = kl_sampler.sample()

    received: list = []
    original = sample._kl_fast_grid_evaluator

    def tracked(resolution, plot_n_modes=None):
        received.append(plot_n_modes)
        return original(resolution, plot_n_modes)

    sample._kl_fast_grid_evaluator = tracked
    plot_function3d_volume_dropdown([sample], resolution=4, plot_n_modes=2, show_plot=False)

    assert received == [2]


def test_plot_n_modes_threaded_through_isosurface(kl_sampler):
    """plot_n_modes flows from plot_function3d_isosurface to the fast path."""
    pytest.importorskip("plotly.graph_objects")
    sample = kl_sampler.sample()

    received: list = []
    original = sample._kl_fast_grid_evaluator

    def tracked(resolution, plot_n_modes=None):
        received.append(plot_n_modes)
        return original(resolution, plot_n_modes)

    sample._kl_fast_grid_evaluator = tracked
    plot_function3d_isosurface(sample, resolution=4, plot_n_modes=2, show_plot=False)

    assert received == [2]


def test_plot_n_modes_threaded_through_axis_slider(kl_sampler):
    """plot_n_modes flows from plot_function3d_axis_slider to the fast path."""
    pytest.importorskip("plotly.graph_objects")
    sample = kl_sampler.sample()

    received: list = []
    original = sample._kl_fast_slice_evaluator

    def tracked(axis, coord, resolution, plot_n_modes=None):
        received.append(plot_n_modes)
        return original(axis, coord, resolution, plot_n_modes)

    sample._kl_fast_slice_evaluator = tracked
    plot_function3d_axis_slider(
        sample, axis='z', n_slices=3, resolution=4, plot_n_modes=2, show_plot=False
    )

    assert len(received) == 3
    assert all(pnm == 2 for pnm in received)
