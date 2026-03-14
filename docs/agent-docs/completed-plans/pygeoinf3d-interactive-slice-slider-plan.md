## Plan: pygeoinf3D Interactive Slice Slider

Add an optional Plotly-based interactive slice viewer for `Function3D` objects on box-like 3D domains. The feature will complement the existing Matplotlib slice API, keep heavy plotting dependencies optional through an `interactive` extra, and demonstrate the new viewer in the least-norm inversion demo notebook.

**Phases 1**
1. **Phase 1: Implement And Validate Interactive Slice Viewer**
    - **Objective:** Add a Plotly slider-based viewer for axis-aligned slices, cover it with focused tests, document the optional backend, and use it in the least-norm demo notebook.
    - **Files/Functions to Modify/Create:** `pygeoinf3D/pygeoinf3d/plot.py`, `pygeoinf3D/tests/test_plot.py`, `pygeoinf3D/pyproject.toml`, `pygeoinf3D/docs/agent-docs/references/living/pygeoinf3d-reference.md`, `pygeoinf3D/demos/least_norm_inversion_demo.ipynb`
    - **Tests to Write:** optional-backend import tests, slider figure structure tests, axis-label/slider-step tests
    - **Steps:**
        1. Add a Plotly-based `plot_function3d_axis_slider` API that supports `x`, `y`, and `z` sliders while keeping Plotly optional.
        2. Add the optional `interactive` extra in `pyproject.toml`, update tests and living reference, and demonstrate the viewer in the notebook.
        3. Run focused plot tests and execute the updated notebook cells to validate the feature.

**Open Questions 2**
1. Plotly will be exposed through an optional `interactive` extra in `pyproject.toml`.
2. The public API name will be `plot_function3d_axis_slider`.
