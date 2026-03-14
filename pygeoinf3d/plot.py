"""Visualization utilities for Function3D objects.

Provides slice-based plotting for checking volumetric kernels and functions.
Static Matplotlib slices are always available; heavier interactive sliders use
Plotly as an optional dependency.
"""

from importlib import import_module
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING, Union, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from pygeoinf3d.core.functions import Function3D

if TYPE_CHECKING:
    import plotly.graph_objects as go


_AXIS_NAMES = ('x', 'y', 'z')


def _parse_axis(axis: Union[int, str]) -> int:
    """Normalize axis input to 0, 1, or 2."""
    if isinstance(axis, (int, np.integer)):
        if axis not in (0, 1, 2):
            raise ValueError(f"Invalid axis index {axis}. Must be 0, 1, or 2.")
        return int(axis)
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


def _validate_n_slices(n_slices: int) -> int:
    """Validate the number of slider positions for interactive plots."""
    if not isinstance(n_slices, (int, np.integer)):
        raise ValueError(
            f"n_slices must be a positive integer, got {n_slices!r}."
        )
    n_slices_int = int(n_slices)
    if n_slices_int <= 0:
        raise ValueError(
            f"n_slices must be a positive integer, got {n_slices_int}."
        )
    return n_slices_int


def _import_plotly_graph_objects() -> Any:
    """Import Plotly lazily so the dependency stays optional."""
    try:
        return import_module('plotly.graph_objects')
    except ImportError:
        raise ImportError(
            "Interactive slice plotting requires the optional 'plotly' "
            "dependency. Install pygeoinf3d[interactive] or pip install "
            "plotly."
        ) from None


def _import_plotly_make_subplots() -> Any:
    """Import plotly.subplots.make_subplots lazily."""
    try:
        return getattr(import_module('plotly.subplots'), 'make_subplots')
    except ImportError:
        raise ImportError(
            "compare_function3d_axis_sliders requires the optional 'plotly' "
            "dependency. Install pygeoinf3d[interactive] or pip install "
            "plotly."
        ) from None


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
    axis_names = _AXIS_NAMES
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


def plot_function3d_axis_slider(
    function: Function3D,
    axis: Union[int, str] = 'z',
    *,
    n_slices: int = 40,
    resolution: Union[int, Tuple[int, int]] = 100,
    show_plot: bool = True,
    title: Optional[str] = None,
    colorscale: str = 'Viridis',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> "go.Figure":
    """Create an interactive Plotly slider for axis-aligned slices.

    Args:
        function: The function to visualize.
        axis: Slice-normal axis ('x', 'y', 'z' or 0, 1, 2).
        n_slices: Number of slider positions spanning the axis bounds.
        resolution: Grid resolution for each 2D slice.
        show_plot: Whether to call ``fig.show()`` before returning.
        title: Figure title. If None, a default title is generated.
        colorscale: Plotly colorscale name.
        vmin: Lower colour scale limit. If omitted, inferred from all slices.
        vmax: Upper colour scale limit. If omitted, inferred from all slices.

    Returns:
        A Plotly ``Figure`` containing a single heatmap trace and slider
        frames.

    Raises:
        ImportError: If Plotly is not installed.
        ValueError: If ``axis``, ``resolution``, or ``n_slices`` is invalid.
    """
    go = _import_plotly_graph_objects()

    axis_idx = _parse_axis(axis)
    axis_names = _AXIS_NAMES
    n_slices_int = _validate_n_slices(n_slices)

    bbox = function.domain.bounding_box()
    low, high = bbox.low, bbox.high
    coords = np.linspace(low[axis_idx], high[axis_idx], n_slices_int)

    u_idx, v_idx = _get_other_axes(axis_idx)
    u_values = None
    v_values = None
    slices: list[np.ndarray] = []

    for coord in coords:
        grid_u, grid_v, values = evaluate_function3d_on_axis_slice(
            function,
            axis=axis_idx,
            coord=float(coord),
            resolution=resolution,
        )
        if u_values is None:
            u_values = grid_u[:, 0]
            v_values = grid_v[0, :]
        slices.append(np.asarray(values.T, dtype=float))

    values_concat = np.concatenate([slice_.ravel() for slice_ in slices])
    zmin = float(values_concat.min()) if vmin is None else float(vmin)
    zmax = float(values_concat.max()) if vmax is None else float(vmax)

    if u_values is None or v_values is None:
        raise RuntimeError("Interactive slice sampling produced no frames.")

    init_idx = n_slices_int // 2
    init_heatmap = go.Heatmap(
        z=slices[init_idx].tolist(),
        x=u_values.tolist(),
        y=v_values.tolist(),
        zmin=zmin,
        zmax=zmax,
        colorscale=colorscale,
        colorbar=dict(title=dict(text='Value', side='right')),
    )

    frames = [
        go.Frame(
            data=[go.Heatmap(z=slice_.tolist())],
            traces=[0],
            name=str(i),
        )
        for i, slice_ in enumerate(slices)
    ]

    if title is None:
        if function.name:
            base_title = function.name
        else:
            base_title = 'Function3D slice viewer'
        title = f"{base_title} ({axis_names[axis_idx]}-axis slider)"

    slider = dict(
        active=init_idx,
        steps=[
            dict(
                args=[
                    [str(i)],
                    {
                        'frame': {'duration': 0, 'redraw': True},
                        'mode': 'immediate',
                        'transition': {'duration': 0},
                    },
                ],
                label=f"{float(coord):.3g}",
                method='animate',
            )
            for i, coord in enumerate(coords)
        ],
        currentvalue=dict(
            prefix=f"{axis_names[axis_idx]} = ",
            visible=True,
            font=dict(size=14),
        ),
        len=0.88,
        x=0.06,
        xanchor='left',
        y=0.0,
        yanchor='top',
        pad=dict(t=10, b=10),
    )

    fig = go.Figure(data=[init_heatmap], frames=frames)
    fig.update_layout(
        title=title,
        xaxis=dict(title=f"{axis_names[u_idx]} coordinate"),
        yaxis=dict(title=f"{axis_names[v_idx]} coordinate"),
        sliders=[slider],
        width=650,
        height=600,
        margin=dict(l=70, r=70, t=70, b=100),
    )
    fig.update_yaxes(scaleanchor='x', scaleratio=1)

    if show_plot:
        fig.show()

    return fig


def compare_function3d_axis_sliders(
    functions: list,
    axis: Union[int, str] = 'z',
    *,
    n_slices: int = 40,
    resolution: Union[int, Tuple[int, int]] = 100,
    show_plot: bool = True,
    titles: Optional[list] = None,
    colorscale: str = 'Viridis',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> "go.Figure":
    """Create a synced multi-panel Plotly slider comparing multiple Function3D objects.

    All panels share a single slider that moves the slice plane along the
    chosen axis. The colour scale is derived globally across all panels and
    all slices, ensuring a consistent visual comparison.

    Args:
        functions: Non-empty list of ``Function3D`` objects to compare. All
            functions must have compatible bounding boxes (same ``low``/``high``
            corners up to floating-point tolerance).
        axis: Slice-normal axis (``'x'``, ``'y'``, ``'z'`` or ``0``/``1``/``2``).
        n_slices: Number of slider positions spanning the full axis extent.
        resolution: Grid resolution for each 2D slice. Positive ``int`` →
            ``(n, n)``; 2-tuple of positive ints → ``(nu, nv)``.
        show_plot: Whether to call ``fig.show()`` before returning.
        titles: Per-panel subplot titles. Must be the same length as
            ``functions`` if provided. Defaults to ``function.name`` or
            ``'Function i'``.
        colorscale: Plotly colorscale name, shared across all panels.
        vmin: Lower colour-scale limit. Inferred from all data if omitted.
        vmax: Upper colour-scale limit. Inferred from all data if omitted.

    Returns:
        A Plotly ``Figure`` with one heatmap per panel and a shared slider.

    Raises:
        ImportError: If Plotly is not installed (install
            ``pygeoinf3d[interactive]``).
        ValueError: If ``functions`` is empty, ``titles`` length mismatches
            ``len(functions)``, bounding boxes are incompatible, or
            ``n_slices``/``resolution`` are invalid.
    """
    go = _import_plotly_graph_objects()
    make_subplots = _import_plotly_make_subplots()

    if not functions:
        raise ValueError("functions must be a non-empty list of Function3D.")

    n_panels = len(functions)

    if titles is not None and len(titles) != n_panels:
        raise ValueError(
            f"titles must have the same length as functions ({n_panels}), "
            f"got {len(titles)}."
        )

    axis_idx = _parse_axis(axis)
    n_slices_int = _validate_n_slices(n_slices)
    axis_names = _AXIS_NAMES

    # Validate that all bounding boxes are compatible
    bbox0 = functions[0].domain.bounding_box()
    for i, fn in enumerate(functions[1:], 1):
        bb = fn.domain.bounding_box()
        if not (np.allclose(bb.low, bbox0.low) and np.allclose(bb.high, bbox0.high)):
            raise ValueError(
                f"All functions must share the same bounding box. "
                f"Function 0 has bbox {bbox0.low}–{bbox0.high}, "
                f"function {i} has {bb.low}–{bb.high}."
            )

    low, high = bbox0.low, bbox0.high
    coords = np.linspace(low[axis_idx], high[axis_idx], n_slices_int)
    u_idx, v_idx = _get_other_axes(axis_idx)

    # Pre-compute all slices — slices_per_fn[fn_idx][slice_idx] is a 2D array
    # transposed to match Plotly's Heatmap (z[row][col] convention).
    u_1d: Optional[np.ndarray] = None
    v_1d: Optional[np.ndarray] = None
    slices_per_fn: list[list[np.ndarray]] = [[] for _ in functions]

    for coord in coords:
        for fn_idx, fn in enumerate(functions):
            grid_u, grid_v, vals = evaluate_function3d_on_axis_slice(
                fn, axis=axis_idx, coord=float(coord), resolution=resolution,
            )
            if u_1d is None:
                u_1d = grid_u[:, 0]
                v_1d = grid_v[0, :]
            slices_per_fn[fn_idx].append(np.asarray(vals.T, dtype=float))

    if u_1d is None or v_1d is None:
        raise RuntimeError("Slice sampling produced no frames.")

    # Per-panel colour limits — each function keeps its own scale so that
    # panels with very different magnitudes (e.g. a model vs. a sensitivity
    # map) all render visibly.  Pass explicit vmin/vmax to lock all panels
    # to the same scale.
    panel_zmins: list[float] = []
    panel_zmaxs: list[float] = []
    for fn_slices in slices_per_fn:
        vals = np.concatenate([s.ravel() for s in fn_slices])
        panel_zmins.append(float(vals.min()) if vmin is None else float(vmin))
        panel_zmaxs.append(float(vals.max()) if vmax is None else float(vmax))

    # Panel titles
    panel_titles = titles if titles is not None else [
        fn.name or f"Function {i}" for i, fn in enumerate(functions)
    ]

    fig = make_subplots(rows=1, cols=n_panels, subplot_titles=panel_titles)

    init_idx = n_slices_int // 2

    # Derive each colorbar's x-position from the actual subplot domain so
    # it sits immediately to the right of its own panel rather than all
    # stacking at the figure's right edge.
    cb_x_positions = []
    for col in range(1, n_panels + 1):
        axis_key = 'xaxis' if col == 1 else f'xaxis{col}'
        domain = getattr(fig.layout, axis_key).domain
        right_edge = float(domain[1]) if domain is not None else col / n_panels
        cb_x_positions.append(right_edge + 0.01)

    for fn_idx in range(n_panels):
        trace = go.Heatmap(
            z=slices_per_fn[fn_idx][init_idx].tolist(),
            x=u_1d.tolist(),
            y=v_1d.tolist(),
            zmin=panel_zmins[fn_idx],
            zmax=panel_zmaxs[fn_idx],
            colorscale=colorscale,
            showscale=True,
            colorbar=dict(
                x=cb_x_positions[fn_idx],
                xanchor='left',
                thickness=14,
                len=0.75,
                y=0.5,
                yanchor='middle',
                title=dict(text='Value', side='right'),
            ),
        )
        fig.add_trace(trace, row=1, col=fn_idx + 1)

    # Each frame updates all n_panels traces simultaneously
    frames = [
        go.Frame(
            data=[
                go.Heatmap(z=slices_per_fn[fn_idx][i].tolist())
                for fn_idx in range(n_panels)
            ],
            traces=list(range(n_panels)),
            name=str(i),
        )
        for i, coord in enumerate(coords)
    ]
    fig.frames = frames

    # Axis labels and equal aspect ratio for each panel
    x_label = f"{axis_names[u_idx]} coordinate"
    y_label = f"{axis_names[v_idx]} coordinate"
    for col in range(1, n_panels + 1):
        fig.update_xaxes(title_text=x_label, row=1, col=col)
        anchor_name = 'x' if col == 1 else f'x{col}'
        fig.update_yaxes(
            title_text=y_label,
            scaleanchor=anchor_name,
            scaleratio=1,
            row=1,
            col=col,
        )

    slider = dict(
        active=init_idx,
        steps=[
            dict(
                args=[
                    [str(i)],
                    {
                        'frame': {'duration': 0, 'redraw': True},
                        'mode': 'immediate',
                        'transition': {'duration': 0},
                    },
                ],
                label=f"{float(coord):.3g}",
                method='animate',
            )
            for i, coord in enumerate(coords)
        ],
        currentvalue=dict(
            prefix=f"{axis_names[axis_idx]} = ",
            visible=True,
            font=dict(size=14),
        ),
        len=0.88,
        x=0.06,
        xanchor='left',
        y=0.0,
        yanchor='top',
        pad=dict(t=10, b=10),
    )

    fig.update_layout(
        sliders=[slider],
        width=min(460 * n_panels + 120, 1600),
        height=520,
        margin=dict(l=70, r=90, t=70, b=100),
    )

    if show_plot:
        fig.show()

    return fig
