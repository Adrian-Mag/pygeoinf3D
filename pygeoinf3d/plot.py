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
from pygeoinf3d.core.paths import Path3D

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


def _validate_plot_n_modes(plot_n_modes: Optional[int]) -> Optional[int]:
    """Validate the optional visualization-only KL mode cap.

    Returns None unchanged. Positive integers are returned as ``int``.
    Zero, negative values, and non-integer types raise ``ValueError``.

    Args:
        plot_n_modes: Desired KL mode cap, or None to use all modes.

    Returns:
        None or the validated positive integer.

    Raises:
        ValueError: If not None and not a positive integer.
    """
    if plot_n_modes is None:
        return None
    if not isinstance(plot_n_modes, (int, np.integer)):
        raise ValueError(
            f"plot_n_modes must be a positive integer or None, "
            f"got {plot_n_modes!r}."
        )
    n = int(plot_n_modes)
    if n < 1:
        raise ValueError(
            f"plot_n_modes must be a positive integer or None, got {n}."
        )
    return n


def _validate_n_samples(n_samples: int) -> int:
    """Validate the number of sample points for path rendering.

    Must be a positive integer >= 2.
    """
    if not isinstance(n_samples, (int, np.integer)):
        raise ValueError(
            f"n_samples must be a positive integer >= 2, got {n_samples!r}."
        )
    n = int(n_samples)
    if n < 2:
        raise ValueError(
            f"n_samples must be a positive integer >= 2, got {n}."
        )
    return n


def _validate_volume_resolution(resolution: int) -> int:
    """Validate the regular 3D grid resolution for volume-style plots."""
    if not isinstance(resolution, (int, np.integer)):
        raise ValueError(
            f"resolution must be an integer >= 2, got {resolution!r}."
        )
    resolution_int = int(resolution)
    if resolution_int < 2:
        raise ValueError(
            f"resolution must be an integer >= 2, got {resolution_int}."
        )
    return resolution_int


def _import_plotly_graph_objects() -> Any:
    """Import Plotly lazily so the dependency stays optional."""
    try:
        return import_module('plotly.graph_objects')
    except ImportError:
        raise ImportError(
            "Interactive plotting requires the optional 'plotly' "
            "dependency. Install pygeoinf3d[interactive] or pip install "
            "plotly."
        ) from None


def _import_plotly_make_subplots() -> Any:
    """Import plotly.subplots.make_subplots lazily."""
    try:
        return getattr(import_module('plotly.subplots'), 'make_subplots')
    except ImportError:
        raise ImportError(
            "Interactive plotting requires the optional 'plotly' "
            "dependency. Install pygeoinf3d[interactive] or pip install "
            "plotly."
        ) from None


def evaluate_function3d_on_axis_slice(
    function: Function3D,
    axis: Union[int, str],
    coord: Optional[float] = None,
    resolution: Union[int, Tuple[int, int]] = 100,
    *,
    plot_n_modes: Optional[int] = None,
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
        plot_n_modes: If the function is a KL sample with tensor-product
            structure, cap the number of KL modes used to this value.
            Silently ignored for generic Function3D objects. Defaults to
            None (use all modes).

    Returns:
        (grid_u, grid_v, values) tuple, where:
        - grid_u: First free axis coordinates, shape (nu, nv).
        - grid_v: Second free axis coordinates, shape (nu, nv).
        - values: Evaluated function values, shape (nu, nv).
    """
    axis_idx = _parse_axis(axis)
    rn_u, rn_v = _validate_resolution(resolution)
    plot_n_modes = _validate_plot_n_modes(plot_n_modes)

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

    # Fast path: KL samples with tensor-product structure bypass the generic
    # point-by-point evaluation in favour of the vectorized einsum path.
    # coord is now always a resolved float (never None).
    fast_path = getattr(function, '_kl_fast_slice_evaluator', None)
    if fast_path is not None:
        return fast_path(axis_idx, coord, (rn_u, rn_v), plot_n_modes)

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


def evaluate_function3d_on_regular_grid(
    function: Function3D,
    resolution: int = 25,
    *,
    plot_n_modes: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample a Function3D on a regular axis-aligned 3D grid.

    Args:
        function: The function to sample.
        resolution: Number of grid points along each axis.
        plot_n_modes: If the function is a KL sample with tensor-product
            structure, cap the number of KL modes used to this value.
            Silently ignored for generic Function3D objects. Defaults to
            None (use all modes).

    Returns:
        ``(X, Y, Z, values)`` where all arrays have shape
        ``(resolution, resolution, resolution)``.
    """
    resolution_int = _validate_volume_resolution(resolution)
    plot_n_modes = _validate_plot_n_modes(plot_n_modes)

    # Fast path: KL samples with tensor-product structure bypass the generic
    # point-by-point evaluation in favour of the vectorized einsum path.
    fast_path = getattr(function, '_kl_fast_grid_evaluator', None)
    if fast_path is not None:
        return fast_path(resolution_int, plot_n_modes)

    bbox = function.domain.bounding_box()
    low, high = bbox.low, bbox.high

    x = np.linspace(low[0], high[0], resolution_int)
    y = np.linspace(low[1], high[1], resolution_int)
    z = np.linspace(low[2], high[2], resolution_int)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    values = np.asarray(function(points), dtype=float).reshape(X.shape)

    return X, Y, Z, values


def plot_function3d_axis_slice(
    function: Function3D,
    axis: Union[int, str],
    coord: Optional[float] = None,
    resolution: Union[int, Tuple[int, int]] = 100,
    show_plot: bool = True,
    title: Optional[str] = None,
    ax: Optional[Axes] = None,
    plot_n_modes: Optional[int] = None,
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
        function, axis=axis_idx, coord=coord, resolution=resolution,
        plot_n_modes=plot_n_modes,
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
    plot_n_modes: Optional[int] = None,
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
            plot_n_modes=plot_n_modes,
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

    if title is None:
        if function.name:
            base_title = function.name
        else:
            base_title = 'Function3D slice viewer'
        title = f"{base_title} ({axis_names[axis_idx]}-axis slider)"

    make_subplots = _import_plotly_make_subplots()
    fig = make_subplots(rows=1, cols=1, subplot_titles=[title])

    fig.add_trace(go.Heatmap(
        z=slices[init_idx].tolist(),
        x=u_values.tolist(),
        y=v_values.tolist(),
        zmin=zmin,
        zmax=zmax,
        colorscale=colorscale,
        colorbar=dict(title=dict(text='Value', side='right')),
    ), row=1, col=1)

    frames = [
        go.Frame(
            data=[go.Heatmap(z=slice_.tolist())],
            traces=[0],
            name=str(i),
        )
        for i, slice_ in enumerate(slices)
    ]
    fig.frames = frames

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
        width=650,
        height=600,
        margin=dict(l=70, r=70, t=70, b=100),
    )
    fig.update_xaxes(
        title_text=f"{axis_names[u_idx]} coordinate",
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title_text=f"{axis_names[v_idx]} coordinate",
        scaleanchor='x',
        scaleratio=1,
        row=1, col=1,
    )

    if show_plot:
        fig.show()

    return fig


def plot_function3d_axis_dropdown_slider(
    functions: list,
    axis: Union[int, str] = 'z',
    *,
    n_slices: int = 20,
    resolution: Union[int, Tuple[int, int]] = 100,
    show_plot: bool = True,
    title: Optional[str] = None,
    labels: Optional[list] = None,
    colorscale: str = 'Viridis',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    per_function_colorscale: bool = True,
    plot_n_modes: Optional[int] = None,
) -> "go.Figure":
    """Interactive heatmap with a dropdown and a shared slice slider.

    Unlike :func:`compare_function3d_axis_sliders` which displays all
    functions side-by-side, this function overlays every function on the same
    axes and exposes a dropdown menu to select which one is visible.  The
    slider moves the slice plane along the chosen axis, and both controls work
    independently.

    Each function gets its own set of slider frames so the colour scale is
    always derived from the selected function's data range.

    Args:
        functions: Non-empty list of ``Function3D`` objects sharing the same
            bounding box.
        axis: Slice-normal axis (``'x'``, ``'y'``, ``'z'`` or
            ``0``/``1``/``2``).
        n_slices: Number of slider positions spanning the full axis extent.
        resolution: Grid resolution for each 2D slice.
        show_plot: Whether to call ``fig.show()`` before returning.
        title: Figure title. If ``None``, the active function label is used.
        labels: Dropdown labels for each function. Defaults to
            ``function.name`` or ``'Function i'``.
        colorscale: Plotly colorscale name.
        vmin: Shared lower colour limit. If omitted, each function uses its
            own data-derived range.
        vmax: Shared upper colour limit. If omitted, each function uses its
            own data-derived range.
        per_function_colorscale: When ``True`` and ``vmin``/``vmax`` are both
            ``None``, each function is colour-scaled independently so that
            low-amplitude functions remain visible. Set ``False`` to share a
            global colour scale.

    Returns:
        A Plotly ``Figure`` with one heatmap trace per (function, slice)
        combination and both dropdown and slider controls.

    Raises:
        ImportError: If Plotly is not installed.
        ValueError: If ``functions`` is empty, bounding boxes mismatch, or
            ``n_slices``/``resolution`` are invalid.
    """
    go = _import_plotly_graph_objects()
    make_subplots = _import_plotly_make_subplots()

    if not functions:
        raise ValueError("functions must be a non-empty list of Function3D.")

    n_fns = len(functions)
    axis_idx = _parse_axis(axis)
    n_slices_int = _validate_n_slices(n_slices)
    axis_names = _AXIS_NAMES
    u_idx, v_idx = _get_other_axes(axis_idx)

    # Validate bounding boxes
    bbox0 = functions[0].domain.bounding_box()
    for i, fn in enumerate(functions[1:], 1):
        bb = fn.domain.bounding_box()
        if not (
            np.allclose(bb.low, bbox0.low)
            and np.allclose(bb.high, bbox0.high)
        ):
            raise ValueError(
                f"All functions must share the same bounding box. "
                f"Function 0 has bbox {bbox0.low}–{bbox0.high}, "
                f"function {i} has {bb.low}–{bb.high}."
            )

    fn_labels = labels if labels is not None else [
        fn.name or f"Function {i + 1}" for i, fn in enumerate(functions)
    ]

    coords = np.linspace(
        float(bbox0.low[axis_idx]),
        float(bbox0.high[axis_idx]),
        n_slices_int,
    )

    # Pre-compute all slices for every function
    # slices_per_fn[fn_idx][slice_idx] → 2D ndarray (transposed for Plotly)
    u_1d: Optional[np.ndarray] = None
    v_1d: Optional[np.ndarray] = None
    slices_per_fn: list[list[np.ndarray]] = [[] for _ in functions]
    zmins: list[float] = []
    zmaxs: list[float] = []

    for fn_idx, fn in enumerate(functions):
        for coord in coords:
            gu, gv, vals = evaluate_function3d_on_axis_slice(
                fn, axis=axis_idx, coord=float(coord),
                resolution=resolution, plot_n_modes=plot_n_modes,
            )
            if u_1d is None:
                u_1d = gu[:, 0]
                v_1d = gv[0, :]
            slices_per_fn[fn_idx].append(np.asarray(vals.T, dtype=float))

        fn_concat = np.concatenate([s.ravel() for s in slices_per_fn[fn_idx]])
        if vmin is not None and vmax is not None:
            zmins.append(float(vmin))
            zmaxs.append(float(vmax))
        elif per_function_colorscale:
            zmins.append(float(fn_concat.min()))
            zmaxs.append(float(fn_concat.max()))
        else:
            # Accumulate globally — filled after all functions
            zmins.append(float(fn_concat.min()))
            zmaxs.append(float(fn_concat.max()))

    if not per_function_colorscale and vmin is None and vmax is None:
        global_min = min(zmins)
        global_max = max(zmaxs)
        zmins = [global_min] * n_fns
        zmaxs = [global_max] * n_fns

    if u_1d is None or v_1d is None:
        raise RuntimeError("Slice sampling produced no frames.")

    init_slice = n_slices_int // 2
    fig = make_subplots(rows=1, cols=1)

    # One heatmap trace per function; only the first is visible initially.
    for fn_idx in range(n_fns):
        fig.add_trace(go.Heatmap(
            z=slices_per_fn[fn_idx][init_slice].tolist(),
            x=u_1d.tolist(),
            y=v_1d.tolist(),
            zmin=zmins[fn_idx],
            zmax=zmaxs[fn_idx],
            colorscale=colorscale,
            colorbar=dict(title=dict(text='Value', side='right')),
            name=fn_labels[fn_idx],
            visible=(fn_idx == 0),
        ), row=1, col=1)

    # n_slices_int frames, each updating ALL traces simultaneously.
    # This means the slider steps never need to change — they always target
    # frames "0".."n_slices_int-1" regardless of which function is active.
    # When a frame fires, every trace gets its new slice data; the visible
    # one shows through, the rest stay hidden.
    frames = []
    for s_idx in range(n_slices_int):
        frames.append(go.Frame(
            data=[go.Heatmap(z=slices_per_fn[fn_idx][s_idx].tolist(),
                             zmin=zmins[fn_idx], zmax=zmaxs[fn_idx])
                  for fn_idx in range(n_fns)],
            traces=list(range(n_fns)),
            name=str(s_idx),
        ))
    fig.frames = frames

    # Fixed slider steps — target frames "0".."n_slices_int-1" for all time.
    slider = dict(
        active=init_slice,
        steps=[
            dict(
                args=[[str(s_idx)],
                      {'frame': {'duration': 0, 'redraw': True},
                       'mode': 'immediate', 'transition': {'duration': 0}}],
                label=f"{float(coords[s_idx]):.3g}",
                method='animate',
            )
            for s_idx in range(n_slices_int)
        ],
        currentvalue=dict(
            prefix=f"{axis_names[axis_idx]} = ",
            visible=True,
            font=dict(size=14),
        ),
        len=0.88, x=0.06, xanchor='left',
        y=0.0,    yanchor='top',
        pad=dict(t=10, b=10),
    )

    # Dropdown: only toggles trace visibility + updates title.
    # No slider step changes — they are fixed and always correct.
    def _dropdown_button(fn_idx: int) -> dict:
        visibility = [j == fn_idx for j in range(n_fns)]
        btn_title = title or fn_labels[fn_idx]
        return dict(
            label=fn_labels[fn_idx],
            method='update',
            args=[
                {'visible': visibility},
                {'title': btn_title},
            ],
        )

    dropdown = dict(
        buttons=[_dropdown_button(i) for i in range(n_fns)],
        direction='down',
        showactive=True,
        x=0.0, xanchor='left',
        y=1.18, yanchor='top',
    )

    initial_title = title or fn_labels[0]
    fig.update_layout(
        title=initial_title,
        updatemenus=[dropdown],
        sliders=[slider],
        width=650,
        height=620,
        margin=dict(l=70, r=70, t=100, b=110),
    )
    fig.update_xaxes(
        title_text=f"{axis_names[u_idx]} coordinate",
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title_text=f"{axis_names[v_idx]} coordinate",
        scaleanchor='x', scaleratio=1,
        row=1, col=1,
    )

    if show_plot:
        fig.show()

    return fig


def plot_function3d_volume_dropdown(
    functions: list[Function3D],
    *,
    resolution: int = 25,
    opacity: float = 0.15,
    surface_count: int = 15,
    colorscale: str = 'RdBu_r',
    robust_percentile: float = 95.0,
    labels: Optional[list[str]] = None,
    show_plot: bool = True,
    title: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    plot_n_modes: Optional[int] = None,
) -> "go.Figure":
    """Create a Plotly volume viewer with a dropdown over functions.

    Each function is sampled internally on the same regular 3D grid, and the
    resulting volume traces are toggled via a Plotly dropdown menu.

    Args:
        functions: Non-empty list of ``Function3D`` objects sharing the same
            bounding box.
        resolution: Number of grid points per axis.
        opacity: Per-trace volume opacity.
        surface_count: Number of rendered isosurfaces inside each volume.
        colorscale: Plotly colorscale name.
        robust_percentile: When ``vmin`` and ``vmax`` are omitted, use a
            symmetric colour range based on this percentile of absolute values.
        labels: Dropdown labels for the functions. Defaults to
            ``function.name`` or ``Function i``.
        show_plot: Whether to call ``fig.show()`` before returning.
        title: Figure title. Defaults to the first label.
        vmin: Lower colour limit. Inferred from data if omitted.
        vmax: Upper colour limit. Inferred from data if omitted.

    Returns:
        A Plotly ``Figure`` containing one ``go.Volume`` trace per function.
    """
    go = _import_plotly_graph_objects()

    if not functions:
        raise ValueError("functions must be a non-empty list of Function3D.")
    if labels is not None and len(labels) != len(functions):
        raise ValueError(
            f"labels must have the same length as functions "
            f"({len(functions)}), "
            f"got {len(labels)}."
        )
    if not (0.0 < float(robust_percentile) <= 100.0):
        raise ValueError(
            "robust_percentile must lie in the interval (0, 100]."
        )

    resolution_int = _validate_volume_resolution(resolution)
    bbox0 = functions[0].domain.bounding_box()
    for i, fn in enumerate(functions[1:], 1):
        bbox = fn.domain.bounding_box()
        if not (
            np.allclose(bbox.low, bbox0.low)
            and np.allclose(bbox.high, bbox0.high)
        ):
            raise ValueError(
                f"All functions must share the same bounding box. "
                f"Function 0 has bbox {bbox0.low}–{bbox0.high}, "
                f"function {i} has {bbox.low}–{bbox.high}."
            )

    plot_labels = labels if labels is not None else [
        fn.name or f"Function {i + 1}" for i, fn in enumerate(functions)
    ]

    X = Y = Z = None
    sampled_values: list[np.ndarray] = []
    for fn in functions:
        Xi, Yi, Zi, values = evaluate_function3d_on_regular_grid(
            fn,
            resolution=resolution_int,
            plot_n_modes=plot_n_modes,
        )
        if X is None:
            X, Y, Z = Xi, Yi, Zi
        sampled_values.append(values)

    if X is None or Y is None or Z is None:
        raise RuntimeError("Volume sampling produced no grid.")

    all_values = np.concatenate([values.ravel() for values in sampled_values])
    if vmin is None and vmax is None:
        cmax = float(np.percentile(np.abs(all_values), robust_percentile))
        used_vmin = -cmax
        used_vmax = cmax
    else:
        used_vmin = float(all_values.min()) if vmin is None else float(vmin)
        used_vmax = float(all_values.max()) if vmax is None else float(vmax)

    x_flat = X.ravel().tolist()
    y_flat = Y.ravel().tolist()
    z_flat = Z.ravel().tolist()

    fig = go.Figure()
    for i, values in enumerate(sampled_values):
        fig.add_trace(go.Volume(
            x=x_flat,
            y=y_flat,
            z=z_flat,
            value=values.ravel().tolist(),
            isomin=used_vmin,
            isomax=used_vmax,
            opacity=opacity,
            surface_count=surface_count,
            colorscale=colorscale,
            cmin=used_vmin,
            cmax=used_vmax,
            showscale=(i == 0),
            colorbar=dict(title='Value', x=1.02),
            name=plot_labels[i],
            visible=(i == 0),
        ))

    buttons = []
    for i, label in enumerate(plot_labels):
        visibility = [j == i for j in range(len(plot_labels))]
        active_title = title or f"{label} — volume rendering"
        buttons.append(dict(
            label=label,
            method='update',
            args=[{'visible': visibility}, {'title': active_title}],
        ))

    initial_title = title or f"{plot_labels[0]} — volume rendering"
    fig.update_layout(
        title=initial_title,
        updatemenus=[dict(
            buttons=buttons,
            direction='down',
            showactive=True,
            x=0.0,
            xanchor='left',
            y=1.15,
            yanchor='top',
        )],
        scene=dict(
            xaxis_title='x',
            yaxis_title='y',
            zaxis_title='z',
            xaxis=dict(range=[float(bbox0.low[0]), float(bbox0.high[0])]),
            yaxis=dict(range=[float(bbox0.low[1]), float(bbox0.high[1])]),
            zaxis=dict(range=[float(bbox0.low[2]), float(bbox0.high[2])]),
            aspectmode='cube',
        ),
        margin=dict(l=0, r=0, b=0, t=80),
        height=600,
    )

    if show_plot:
        fig.show()

    return fig


def plot_function3d_isosurface(
    function: Function3D,
    *,
    resolution: int = 30,
    n_levels: int = 3,
    sigma: float = 1.0,
    opacity: float = 0.5,
    colorscale: str = "RdBu_r",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    show_plot: bool = True,
    title: Optional[str] = None,
    plot_n_modes: Optional[int] = None,
) -> "go.Figure":
    """Create an interactive 3D isosurface plot of a Function3D.

    Evaluates the function on a regular 3D grid and renders isosurfaces
    using Plotly's ``go.Isosurface`` trace.  By default the iso-level range
    spans ±``sigma`` × standard deviation of the sampled values, which
    works naturally for zero-mean Gaussian-prior samples.

    Args:
        function: The function to visualize.
        resolution: Number of grid points per axis (``resolution³`` total
            evaluation points).  Defaults to 30.
        n_levels: Number of iso-levels rendered between the colour limits.
            Defaults to 3.
        sigma: When ``vmin``/``vmax`` are not supplied, the colour range is
            set to ±``sigma`` × std of the sampled values.  Defaults to 1.0.
        opacity: Opacity of each isosurface (0–1).  Defaults to 0.5.
        colorscale: Plotly named colorscale.  Use a diverging scale such as
            ``"RdBu_r"`` for zero-mean fields.  Defaults to ``"RdBu_r"``.
        vmin: Lower iso-level bound.  Inferred from data if omitted.
        vmax: Upper iso-level bound.  Inferred from data if omitted.
        show_plot: Whether to call ``fig.show()`` before returning.
        title: Figure title.  Defaults to ``function.name`` or a generic
            description.

    Returns:
        A Plotly ``Figure`` containing a single ``go.Isosurface`` trace.

    Raises:
        ImportError: If Plotly is not installed.
        ValueError: If ``resolution < 2`` or ``n_levels < 1``.
    """
    go = _import_plotly_graph_objects()

    resolution_int = _validate_volume_resolution(resolution)
    if n_levels < 1:
        raise ValueError(f"n_levels must be >= 1, got {n_levels}.")

    X, Y, Z, values = evaluate_function3d_on_regular_grid(
        function,
        resolution=resolution_int,
        plot_n_modes=plot_n_modes,
    )

    std = float(values.std())
    used_vmin = float(vmin) if vmin is not None else -sigma * std
    used_vmax = float(vmax) if vmax is not None else sigma * std

    bbox = function.domain.bounding_box()
    low, high = bbox.low, bbox.high

    if title is None:
        title = (function.name or "Function3D") + " — isosurface"

    fig = go.Figure(go.Isosurface(
        x=X.ravel().tolist(),
        y=Y.ravel().tolist(),
        z=Z.ravel().tolist(),
        value=values.ravel().tolist(),
        isomin=used_vmin,
        isomax=used_vmax,
        surface_count=n_levels,
        opacity=opacity,
        colorscale=colorscale,
        cmin=used_vmin,
        cmax=used_vmax,
        caps=dict(x_show=False, y_show=False, z_show=False),
        colorbar=dict(title="value"),
    ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="z",
            xaxis=dict(range=[low[0], high[0]]),
            yaxis=dict(range=[low[1], high[1]]),
            zaxis=dict(range=[low[2], high[2]]),
            aspectmode="cube",
        ),
        height=600,
        margin=dict(l=0, r=0, b=0, t=60),
    )

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
    plot_n_modes: Optional[int] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> "go.Figure":
    """Create a synced multi-panel Plotly slider.

    All panels share a single slider that moves the slice plane along the
    chosen axis. The colour scale is derived globally across all panels and
    all slices, ensuring a consistent visual comparison.

    Args:
        functions: Non-empty list of ``Function3D`` objects to compare. All
            functions must have compatible bounding boxes.
        axis: Slice-normal axis (``'x'``, ``'y'``, ``'z'`` or
            ``0``/``1``/``2``).
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
        if not (
            np.allclose(bb.low, bbox0.low)
            and np.allclose(bb.high, bbox0.high)
        ):
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
                plot_n_modes=plot_n_modes,
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
    # stacking at the figure's right edge.  Clamp to 1.0 to avoid the last
    # panel's colorbar being cut off when the figure has limited right margin.
    cb_x_positions = []
    for col in range(1, n_panels + 1):
        axis_key = 'xaxis' if col == 1 else f'xaxis{col}'
        domain = getattr(fig.layout, axis_key).domain
        right_edge = float(domain[1]) if domain is not None else col / n_panels
        cb_x_positions.append(min(right_edge + 0.01, 1.0))

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
        width=width if width is not None else min(460 * n_panels + 120, 1600),
        height=height if height is not None else 520,
        margin=dict(l=70, r=90, t=70, b=100),
    )

    if show_plot:
        fig.show()

    return fig


def compare_panels_axis_dropdown_sliders(
    panel_functions: list,
    axis: Union[int, str] = 'z',
    *,
    n_slices: int = 20,
    resolution: Union[int, Tuple[int, int]] = 100,
    show_plot: bool = True,
    panel_labels: Optional[list] = None,
    panel_titles: Optional[list] = None,
    colorscale: str = 'Viridis',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    plot_n_modes: Optional[int] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> "go.Figure":
    """Side-by-side panels with a shared slice slider and per-panel dropdowns.

    Each panel displays one function chosen from its own candidate list via an
    independent dropdown.  All panels share a single slider that advances the
    slice plane along the chosen axis.  Colour limits are derived per-panel by
    default; pass ``vmin``/``vmax`` to lock all panels to a common scale.

    The per-panel dropdowns use Plotly's ``restyle`` method so that changing
    the selection in one panel has no effect on the other panels' visibility.

    Args:
        panel_functions: Non-empty list of panels.  Each panel is itself a
            non-empty list of ``Function3D`` objects — the candidates for that
            panel's dropdown.  All functions across all panels must share the
            same bounding box.
        axis: Slice-normal axis (``'x'``, ``'y'``, ``'z'`` or
            ``0``/``1``/``2``).
        n_slices: Number of slider positions spanning the full axis extent.
        resolution: Grid resolution for each 2D slice.
        show_plot: Whether to call ``fig.show()`` before returning.
        panel_labels: Per-panel dropdown labels.  Must be a list of lists
            with the same nesting shape as ``panel_functions``.  Defaults to
            ``function.name`` or ``'Function i'``.
        panel_titles: Subplot title for each panel.  Must have the same length
            as ``panel_functions`` if provided.
        colorscale: Plotly colorscale name, applied to every trace.
        vmin: Global lower colour limit.  Per-panel limits are used if omitted.
        vmax: Global upper colour limit.  Per-panel limits are used if omitted.
        plot_n_modes: Optional KL mode cap forwarded to
            :func:`evaluate_function3d_on_axis_slice`.
        width: Figure width in pixels.  Defaults to ``min(460 * n_panels + 120,
            1600)``.
        height: Figure height in pixels.  Defaults to 520.

    Returns:
        A Plotly ``Figure`` with one heatmap trace per (panel, candidate
        function), a single shared slider, and one dropdown per panel.

    Raises:
        ImportError: If Plotly is not installed.
        ValueError: If ``panel_functions`` is empty, any panel is empty,
            ``panel_labels`` has a length mismatch, bounding boxes are
            inconsistent, or ``n_slices``/``resolution`` are invalid.
    """
    go = _import_plotly_graph_objects()
    make_subplots = _import_plotly_make_subplots()

    # --- Input validation ---
    if not panel_functions:
        raise ValueError("panel_functions must be a non-empty list of panels.")

    n_panels = len(panel_functions)

    for p, panel in enumerate(panel_functions):
        if not panel:
            raise ValueError(
                f"Panel {p} is empty. Each panel must contain at least one "
                f"Function3D."
            )

    if panel_labels is not None:
        if len(panel_labels) != n_panels:
            raise ValueError(
                f"panel_labels must have the same length as panel_functions "
                f"({n_panels}), got {len(panel_labels)}."
            )
        for p, (panel, plabels) in enumerate(
            zip(panel_functions, panel_labels)
        ):
            if len(plabels) != len(panel):
                raise ValueError(
                    f"panel_labels[{p}] must have the same length as "
                    f"panel_functions[{p}] ({len(panel)}), "
                    f"got {len(plabels)}."
                )

    if panel_titles is not None and len(panel_titles) != n_panels:
        raise ValueError(
            f"panel_titles must have the same length as panel_functions "
            f"({n_panels}), got {len(panel_titles)}."
        )

    # Validate bounding boxes — all functions across all panels must match
    bbox0 = panel_functions[0][0].domain.bounding_box()
    for p, panel in enumerate(panel_functions):
        for fn_i, fn in enumerate(panel):
            bb = fn.domain.bounding_box()
            if not (
                np.allclose(bb.low, bbox0.low)
                and np.allclose(bb.high, bbox0.high)
            ):
                raise ValueError(
                    f"All functions must share the same bounding box. "
                    f"Reference {bbox0.low}–{bbox0.high}, "
                    f"panel {p} function {fn_i} has {bb.low}–{bb.high}."
                )

    axis_idx = _parse_axis(axis)
    n_slices_int = _validate_n_slices(n_slices)
    axis_names = _AXIS_NAMES
    u_idx, v_idx = _get_other_axes(axis_idx)

    low, high = bbox0.low, bbox0.high
    coords = np.linspace(low[axis_idx], high[axis_idx], n_slices_int)

    # Build effective per-panel per-function labels
    if panel_labels is not None:
        effective_labels: list[list[str]] = panel_labels
    else:
        effective_labels = [
            [
                fn.name or f"Function {fn_i + 1}"
                for fn_i, fn in enumerate(panel)
            ]
            for panel in panel_functions
        ]

    # Subplot titles
    if panel_titles is not None:
        subplot_titles = panel_titles
    else:
        subplot_titles = [
            effective_labels[p][0] if len(effective_labels[p]) == 1
            else f"Panel {p + 1}"
            for p in range(n_panels)
        ]

    # Pre-compute slices
    # slices_data[p][fn_i] stores n_slices 2D arrays for Plotly heatmaps.
    u_1d: Optional[np.ndarray] = None
    v_1d: Optional[np.ndarray] = None
    slices_data: list[list[list[np.ndarray]]] = []
    panel_zmins: list[list[float]] = []
    panel_zmaxs: list[list[float]] = []

    for p, panel in enumerate(panel_functions):
        panel_slices: list[list[np.ndarray]] = [[] for _ in panel]
        for fn_i, fn in enumerate(panel):
            for coord in coords:
                gu, gv, vals = evaluate_function3d_on_axis_slice(
                    fn, axis=axis_idx, coord=float(coord),
                    resolution=resolution, plot_n_modes=plot_n_modes,
                )
                if u_1d is None:
                    u_1d = gu[:, 0]
                    v_1d = gv[0, :]
                panel_slices[fn_i].append(np.asarray(vals.T, dtype=float))

        slices_data.append(panel_slices)

        fn_zmins: list[float] = []
        fn_zmaxs: list[float] = []
        for fn_i in range(len(panel)):
            fn_all = np.concatenate([s.ravel() for s in panel_slices[fn_i]])
            fn_zmins.append(
                float(vmin) if vmin is not None else float(fn_all.min())
            )
            fn_zmaxs.append(
                float(vmax) if vmax is not None else float(fn_all.max())
            )
        panel_zmins.append(fn_zmins)
        panel_zmaxs.append(fn_zmaxs)

    if u_1d is None or v_1d is None:
        raise RuntimeError("Slice sampling produced no frames.")

    init_slice = n_slices_int // 2

    fig = make_subplots(rows=1, cols=n_panels, subplot_titles=subplot_titles)

    # Compute trace start index for each panel
    panel_offsets: list[int] = [0]
    for panel in panel_functions[:-1]:
        panel_offsets.append(panel_offsets[-1] + len(panel))

    total_traces = sum(len(p) for p in panel_functions)

    # Colorbar x-positions — one per panel, just right of that panel's domain.
    # Clamp to 1.0 to prevent the last panel's colorbar from being clipped
    # when the figure right margin is narrow.
    cb_x_positions: list[float] = []
    for col in range(1, n_panels + 1):
        axis_key = 'xaxis' if col == 1 else f'xaxis{col}'
        domain = getattr(fig.layout, axis_key).domain
        right_edge = float(domain[1]) if domain is not None else col / n_panels
        cb_x_positions.append(min(right_edge + 0.01, 1.0))

    # --- Add traces (one per panel × function) ---
    for p, panel in enumerate(panel_functions):
        for fn_i in range(len(panel)):
            fig.add_trace(
                go.Heatmap(
                    z=slices_data[p][fn_i][init_slice].tolist(),
                    x=u_1d.tolist(),
                    y=v_1d.tolist(),
                    zmin=panel_zmins[p][fn_i],
                    zmax=panel_zmaxs[p][fn_i],
                    colorscale=colorscale,
                    showscale=True,
                    colorbar=dict(
                        x=cb_x_positions[p],
                        xanchor='left',
                        thickness=14,
                        len=0.75,
                        y=0.5,
                        yanchor='middle',
                        title=dict(text='Value', side='right'),
                    ),
                    name=effective_labels[p][fn_i],
                    visible=(fn_i == 0),
                ),
                row=1, col=p + 1,
            )

    # --- Frames — update every trace on every slider step ---
    trace_order = list(range(total_traces))
    frames = []
    for s_i in range(n_slices_int):
        frame_data = []
        for p, panel in enumerate(panel_functions):
            for fn_i in range(len(panel)):
                frame_data.append(go.Heatmap(
                    z=slices_data[p][fn_i][s_i].tolist(),
                    zmin=panel_zmins[p][fn_i],
                    zmax=panel_zmaxs[p][fn_i],
                ))
        frames.append(go.Frame(
            data=frame_data,
            traces=trace_order,
            name=str(s_i),
        ))
    fig.frames = frames

    # --- Shared slider ---
    slider = dict(
        active=init_slice,
        steps=[
            dict(
                args=[
                    [str(s_i)],
                    {
                        'frame': {'duration': 0, 'redraw': True},
                        'mode': 'immediate',
                        'transition': {'duration': 0},
                    },
                ],
                label=f"{float(coords[s_i]):.3g}",
                method='animate',
            )
            for s_i in range(n_slices_int)
        ],
        currentvalue=dict(
            prefix=f"{axis_names[axis_idx]} = ",
            visible=True,
            font=dict(size=14),
        ),
        len=0.88, x=0.06, xanchor='left',
        y=0.0, yanchor='top',
        pad=dict(t=10, b=10),
    )

    # --- Per-panel dropdowns using restyle ---
    # restyle only alters the listed trace indices for the chosen panel.
    updatemenus = []
    for p, panel in enumerate(panel_functions):
        panel_trace_range = list(
            range(panel_offsets[p], panel_offsets[p] + len(panel))
        )
        buttons = []
        for fn_i in range(len(panel)):
            visibility = [fn_i == j for j in range(len(panel))]
            buttons.append(dict(
                label=effective_labels[p][fn_i],
                method='restyle',
                args=[{'visible': visibility}, panel_trace_range],
            ))

        # Position each dropdown above its panel using the subplot x-domain
        col = p + 1
        axis_key = 'xaxis' if col == 1 else f'xaxis{col}'
        domain = getattr(fig.layout, axis_key).domain
        x_left = float(domain[0]) if domain is not None else p / n_panels

        updatemenus.append(dict(
            buttons=buttons,
            direction='down',
            showactive=True,
            x=x_left,
            xanchor='left',
            y=1.18,
            yanchor='top',
        ))

    # --- Axis labels ---
    x_label = f"{axis_names[u_idx]} coordinate"
    y_label = f"{axis_names[v_idx]} coordinate"
    for col in range(1, n_panels + 1):
        fig.update_xaxes(title_text=x_label, row=1, col=col)
        anchor_name = 'x' if col == 1 else f'x{col}'
        fig.update_yaxes(
            title_text=y_label,
            scaleanchor=anchor_name,
            scaleratio=1,
            row=1, col=col,
        )

    fig.update_layout(
        updatemenus=updatemenus,
        sliders=[slider],
        width=width if width is not None else min(460 * n_panels + 120, 1600),
        height=height if height is not None else 520,
        margin=dict(l=70, r=90, t=100, b=120),
    )

    if show_plot:
        fig.show()

    return fig


def _bbox_wireframe_trace(go: Any, low: np.ndarray, high: np.ndarray) -> Any:
    """Return a ``go.Scatter3d`` wireframe trace for an axis-aligned box.

    The 12 edges of the box are joined into a single line trace using
    ``None`` separators.

    Args:
        go: The ``plotly.graph_objects`` module (already imported).
        low:  Shape ``(3,)`` lower corner of the box.
        high: Shape ``(3,)`` upper corner of the box.

    Returns:
        A ``go.Scatter3d`` trace.
    """
    lx, ly, lz = float(low[0]), float(low[1]), float(low[2])
    hx, hy, hz = float(high[0]), float(high[1]), float(high[2])

    # 12 edges: 4 bottom, 4 top, 4 vertical
    edges = [
        ([lx, hx], [ly, ly], [lz, lz]),
        ([hx, hx], [ly, hy], [lz, lz]),
        ([hx, lx], [hy, hy], [lz, lz]),
        ([lx, lx], [hy, ly], [lz, lz]),
        ([lx, hx], [ly, ly], [hz, hz]),
        ([hx, hx], [ly, hy], [hz, hz]),
        ([hx, lx], [hy, hy], [hz, hz]),
        ([lx, lx], [hy, ly], [hz, hz]),
        ([lx, lx], [ly, ly], [lz, hz]),
        ([hx, hx], [ly, ly], [lz, hz]),
        ([hx, hx], [hy, hy], [lz, hz]),
        ([lx, lx], [hy, hy], [lz, hz]),
    ]
    xs: list = []
    ys: list = []
    zs: list = []
    for ex, ey, ez in edges:
        xs.extend(ex + [None])
        ys.extend(ey + [None])
        zs.extend(ez + [None])

    return go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode='lines',
        line=dict(color='grey', width=1, dash='dash'),
        name='domain box',
        showlegend=True,
    )


def plot_paths3d_interactive(
    paths: Union["Path3D", list],
    *,
    n_samples: int = 100,
    show_domain_box: bool = False,
    show_markers: bool = True,
    labels: Optional[list] = None,
    show_plot: bool = True,
    title: Optional[str] = None,
) -> "go.Figure":
    """Create an interactive Plotly 3D line plot.

    The plot renders one or more ``Path3D`` objects.

    Each path is sampled uniformly in normalized arc-length and rendered as a
    ``go.Scatter3d`` line trace.  Optionally, start/end markers and a domain
    bounding-box wireframe are overlaid.

    Args:
        paths: A single ``Path3D`` or a list of ``Path3D`` objects.
        n_samples: Number of equally-spaced arc-length nodes used to render
            each path.  Must be >= 2.  Defaults to 100.
        show_domain_box: If *True* and the first path has a ``domain`` with a
            ``bounding_box`` method, overlay it as a dashed wireframe.
            Defaults to *False*.
        show_markers: If *True*, add start (circle) and end (diamond) marker
            traces at ``t=0`` and ``t=1`` for each path.  Defaults to *True*.
        labels: Optional list of trace labels (one per path).  Defaults to
            ``'Path 1'``, ``'Path 2'``, ...
        show_plot: Whether to call ``fig.show()`` before returning.
            Defaults to *True*.
        title: Figure title.  Defaults to a generic description.

    Returns:
        A Plotly ``Figure`` with one ``go.Scatter3d`` line trace per path
        (plus optional start/end marker traces and an optional domain-box
        wireframe).

    Raises:
        ImportError: If Plotly is not installed.
        ValueError: If ``n_samples < 2`` or ``labels`` has the wrong length.
    """
    go = _import_plotly_graph_objects()
    n_samples_int = _validate_n_samples(n_samples)

    # Normalise single path to a list
    if isinstance(paths, Path3D):
        path_list: list = [paths]
    else:
        path_list = list(paths)

    if not path_list:
        raise ValueError("paths must be a non-empty Path3D or list of Path3D.")

    for idx, item in enumerate(path_list):
        if not isinstance(item, Path3D):
            raise ValueError(
                f"Every element of paths must be a Path3D instance. "
                f"Got {type(item)!r} at index {idx}."
            )

    if labels is not None and len(labels) != len(path_list):
        raise ValueError(
            f"labels must have the same length as paths "
            f"({len(path_list)}), got {len(labels)}."
        )

    plot_labels = labels if labels is not None else [
        f"Path {i + 1}" for i in range(len(path_list))
    ]

    fig = go.Figure()

    for i, path in enumerate(path_list):
        _, pts = path.arc_length_nodes(n_samples_int)   # (n, 3)
        xs = pts[:, 0].tolist()
        ys = pts[:, 1].tolist()
        zs = pts[:, 2].tolist()

        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode='lines',
            name=plot_labels[i],
            line=dict(width=4),
        ))

        if show_markers:
            fig.add_trace(go.Scatter3d(
                x=[xs[0]], y=[ys[0]], z=[zs[0]],
                mode='markers',
                marker=dict(size=6, symbol='circle'),
                name=f"{plot_labels[i]} start",
                showlegend=False,
            ))
            fig.add_trace(go.Scatter3d(
                x=[xs[-1]], y=[ys[-1]], z=[zs[-1]],
                mode='markers',
                marker=dict(size=6, symbol='diamond'),
                name=f"{plot_labels[i]} end",
                showlegend=False,
            ))

    if show_domain_box:
        domain = path_list[0].domain
        if domain is not None and hasattr(domain, 'bounding_box'):
            bbox = domain.bounding_box()
            fig.add_trace(_bbox_wireframe_trace(go, bbox.low, bbox.high))

    if title is None:
        n = len(path_list)
        title = f"Path3D viewer ({n} path{'s' if n != 1 else ''})"

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='x',
            yaxis_title='y',
            zaxis_title='z',
            aspectmode='data',
        ),
        height=600,
        margin=dict(l=0, r=0, b=0, t=60),
    )

    if show_plot:
        fig.show()

    return fig
