"""Visualization utilities for Function3D objects.

Provides slice-based plotting for checking volumetric kernels and functions.
"""

from typing import Any, Dict, Optional, Tuple, Union, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from pygeoinf3d.core.functions import Function3D


def _parse_axis(axis: Union[int, str]) -> int:
    """Normalize axis input to 0, 1, or 2."""
    if isinstance(axis, int):
        if axis not in (0, 1, 2):
            raise ValueError(f"Invalid axis index {axis}. Must be 0, 1, or 2.")
        return axis
    s = str(axis).lower()
    mapping = {'x': 0, 'y': 1, 'z': 2}
    if s not in mapping:
        raise ValueError(f"Invalid axis '{axis}'. Must be 'x', 'y', or 'z'.")
    return mapping[s]


def _get_other_axes(axis_idx: int) -> Tuple[int, int]:
    """Return indices of the two orthogonal axes in standard order.

    - z (2) -> (x, y) = (0, 1)
    - y (1) -> (x, z) = (0, 2)
    - x (0) -> (y, z) = (1, 2)
    """
    if axis_idx == 0:
        return (1, 2)
    elif axis_idx == 1:
        return (0, 2)
    return (0, 1)


def _validate_resolution(
    resolution: Union[int, Tuple[int, int]],
) -> Tuple[int, int]:
    """Parse and validate resolution into a (nu, nv) pair of positive ints.

    Raises:
        ValueError: If resolution is not a positive int or a valid 2-tuple.
    """
    if isinstance(resolution, (int, np.integer)):
        n = int(resolution)
        if n <= 0:
            raise ValueError(
                f"Resolution must be a positive int, got {n}."
            )
        return n, n
    try:
        items = tuple(resolution)  # type: ignore[arg-type]
    except TypeError:
        raise ValueError(
            "Resolution must be a positive int or a 2-tuple of "
            f"positive ints, got {resolution!r}."
        ) from None
    if len(items) != 2:
        raise ValueError(
            f"Tuple resolution must have exactly 2 elements, "
            f"got {len(items)}."
        )
    nu, nv = int(items[0]), int(items[1])
    if nu <= 0 or nv <= 0:
        raise ValueError(
            "Tuple resolution values must be positive ints, "
            f"got ({nu}, {nv})."
        )
    return nu, nv


def evaluate_function3d_on_axis_slice(
    function: Function3D,
    axis: Union[int, str],
    coord: Optional[float] = None,
    resolution: Union[int, Tuple[int, int]] = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample a Function3D on a 2D slice aligned with an axis.

    The grid covers the full extent of the domain's bounding box in the
    two free dimensions.

    Args:
        function: The function to sample.
        axis: The fixed axis ('x', 'y', 'z' or 0, 1, 2).
        coord: The coordinate value along the fixed axis. If None, uses the
            midpoint of the domain's bounding box.
        resolution: Number of sample points along the two free axes.
            If int, uses (n, n). If tuple of ints, uses (n_u, n_v).

    Returns:
        (grid_u, grid_v, values) tuple, where:
        - grid_u: First free axis coordinates, shape (nu, nv).
        - grid_v: Second free axis coordinates, shape (nu, nv).
        - values: Evaluated function values, shape (nu, nv).
    """
    axis_idx = _parse_axis(axis)
    rn_u, rn_v = _validate_resolution(resolution)

    bbox = function.domain.bounding_box()
    low, high = bbox.low, bbox.high

    if coord is None:
        coord = 0.5 * (low[axis_idx] + high[axis_idx])
    else:
        eps = 1e-9 * max(1.0, high[axis_idx] - low[axis_idx])
        if not (low[axis_idx] - eps <= coord <= high[axis_idx] + eps):
            raise ValueError(
                f"Slice coordinate {coord} is outside domain bounds "
                f"[{low[axis_idx]}, {high[axis_idx]}] for axis {axis_idx}."
            )

    u_idx, v_idx = _get_other_axes(axis_idx)
    u = np.linspace(low[u_idx], high[u_idx], rn_u)
    v = np.linspace(low[v_idx], high[v_idx], rn_v)
    grid_u, grid_v = np.meshgrid(u, v, indexing='ij')

    flat_u = grid_u.ravel()
    flat_v = grid_v.ravel()
    n_points = flat_u.size
    points = np.zeros((n_points, 3))
    points[:, axis_idx] = coord
    points[:, u_idx] = flat_u
    points[:, v_idx] = flat_v

    # Coerce to ndarray: function() may return float for single-point batches
    values_flat = np.asarray(function(points))
    values = values_flat.reshape(grid_u.shape)

    return grid_u, grid_v, values


def plot_function3d_axis_slice(
    function: Function3D,
    axis: Union[int, str],
    coord: Optional[float] = None,
    resolution: Union[int, Tuple[int, int]] = 100,
    show_plot: bool = True,
    title: Optional[str] = None,
    ax: Optional[Axes] = None,
    **kwargs: Any,
) -> Tuple[Figure, Axes, Dict[str, Any]]:
    """Plot a heatmap of a Function3D on an axis-aligned slice.

    Args:
        function: The function to visualize.
        axis: The slice axis ('x', 'y', 'z' or 0, 1, 2).
        coord: Coordinate along the slice axis (default: center of domain).
        resolution: Grid resolution for sampling.
        show_plot: Whether to call plt.show() (default: True).
        title: Plot title. If None, generates a default description.
        ax: Existing axes to plot on. If None, creates a new figure.
        **kwargs: Additional arguments passed to plt.pcolormesh.

    Returns:
        (fig, ax, payload) tuple, where payload contains intermediate data:
        {'grid_u', 'grid_v', 'values', 'axis_idx', 'coord'}.
    """
    axis_idx = _parse_axis(axis)
    axis_names = ['x', 'y', 'z']
    u_idx, v_idx = _get_other_axes(axis_idx)

    grid_u, grid_v, values = evaluate_function3d_on_axis_slice(
        function, axis=axis_idx, coord=coord, resolution=resolution
    )

    # Determine the used coord if it was None
    used_coord = coord
    if used_coord is None:
        bbox = function.domain.bounding_box()
        used_coord = 0.5 * (bbox.low[axis_idx] + bbox.high[axis_idx])

    if ax is None:
        fig, ax = plt.subplots(constrained_layout=True)
    else:
        fig = cast(Figure, ax.figure)

    # shading='auto' handles mismatched X/Y/C shapes in newer matplotlib
    mesh = ax.pcolormesh(grid_u, grid_v, values, shading='auto', **kwargs)

    if title is None:
        slice_desc = f"{axis_names[axis_idx]}={used_coord:.3g}"
        if function.name:
            title = f"{function.name} ({slice_desc})"
        else:
            title = f"Slice {slice_desc}"

    ax.set_title(title)
    ax.set_xlabel(f"{axis_names[u_idx]} coordinate")
    ax.set_ylabel(f"{axis_names[v_idx]} coordinate")
    ax.set_aspect('equal', adjustable='box')

    # Add colorbar
    fig.colorbar(mesh, ax=ax, label='Value')

    if show_plot:
        plt.show()

    payload: Dict[str, Any] = {
        'grid_u': grid_u,
        'grid_v': grid_v,
        'values': values,
        'axis_idx': axis_idx,
        'coord': used_coord,
    }

    return fig, ax, payload
