import matplotlib.pyplot as plt
import numpy as np
import pytest

from pygeoinf3d.core.functions import Function3D
from pygeoinf3d.geometries.box.domain import BoxDomain
from pygeoinf3d.plot import (
    evaluate_function3d_on_axis_slice,
    plot_function3d_axis_slice,
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
