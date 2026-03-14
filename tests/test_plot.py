import matplotlib.pyplot as plt
import numpy as np
import pytest
from typing import Any, cast

import pygeoinf3d.plot as plot_module

from pygeoinf3d.core.functions import Function3D
from pygeoinf3d.geometries.box.domain import BoxDomain
from pygeoinf3d.plot import (
    evaluate_function3d_on_axis_slice,
    plot_function3d_axis_slice,
    plot_function3d_axis_slider,
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
    """Each panel gets its own colour limits; different-magnitude functions stay visible."""
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
    assert zmaxs[1] > 10 * zmaxs[0], "Panel 2 (f2) should have a much larger zmax than panel 1 (f1)"


def test_compare_function3d_axis_sliders_global_color_limits_via_vmin_vmax(two_functions):
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


def test_compare_function3d_axis_sliders_validation_titles_length(two_functions):
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

    other_domain = BoxDomain(np.array([0.0, 0.0, 0.0]), np.array([2.0, 2.0, 2.0]))

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


def test_compare_function3d_axis_sliders_requires_plotly(two_functions, monkeypatch):
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
