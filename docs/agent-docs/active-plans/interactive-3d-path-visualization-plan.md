## Plan: Interactive 3D Path Visualization

Add a reusable Plotly-based 3D path viewer to pygeoinf3D, then use it in the path-forms notebook so path geometry is shown interactively in 3D rather than only with 2D projections. The implementation will support both single-path and multi-path input from the start, keep the domain box optional, and replace rather than duplicate the earlier 2D-only path-view emphasis in the notebook.

**Phases 3**
1. **Phase 1: Add reusable Path3D plotting API**
    - **Objective:** Introduce a reusable interactive 3D plotting helper for one or more `Path3D` objects in the package visualization module.
    - **Files/Functions to Modify/Create:** `pygeoinf3d/plot.py`; `tests/test_plot.py`; `pygeoinf3d/__init__.py`
    - **Tests to Write:** `test_plot_path3d_interactive_smoke`; `test_plot_paths3d_interactive_multiple_paths`; `test_plot_paths3d_interactive_requires_plotly`; `test_plot_paths3d_interactive_validates_n_samples`
    - **Steps:**
        1. Write failing tests for a Plotly `Scatter3d`-based viewer covering single and multiple paths.
        2. Implement a plotting helper that samples `Path3D` objects, renders optional start/end markers, and can optionally overlay a domain bounding box.
        3. Export the helper through the public package surface and rerun the targeted plot tests.
2. **Phase 2: Upgrade the path-forms notebook demo**
    - **Objective:** Replace the notebook's current path-visualization cells with interactive 3D visuals built on the new plotting helper.
    - **Files/Functions to Modify/Create:** `demos/7_path_forms_demo.ipynb`
    - **Tests to Write:** Execute the affected notebook cells after editing to confirm the new interactive figures render.
    - **Steps:**
        1. Update notebook imports to use the new path plotting helper.
        2. Replace the early path-construction plot with an interactive 3D path view.
        3. Replace the measurement-path overview with an interactive multi-path 3D figure.
3. **Phase 3: Sync references**
    - **Objective:** Update package living references to describe the new reusable path plotting API and notebook usage.
    - **Files/Functions to Modify/Create:** `docs/agent-docs/references/living/pygeoinf3d-reference.md`
    - **Tests to Write:** None.
    - **Steps:**
        1. Update the visualization capabilities summary in the living reference.
        2. Note that `7_path_forms_demo.ipynb` now uses interactive 3D path rendering.
        3. Verify the reference remains consistent with the current package status.

**Open Questions 3**
1. None. The user confirmed support for both single and multiple paths from the start.
2. None. The user confirmed the domain box should be shown only when requested.
3. None. The user confirmed the notebook should not retain the earlier 2D path-view emphasis.