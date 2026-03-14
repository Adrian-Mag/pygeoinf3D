## Phase 1 Complete: Implement And Validate Interactive Slice Viewer

Added an optional Plotly-based interactive slice slider for `Function3D` objects in `pygeoinf3d.plot`, supporting `x`, `y`, and `z` traversal through box domains while keeping the existing Matplotlib slice API unchanged. The feature is covered by focused tests, documented in the living reference, and used in the least-norm demo notebook with a graceful fallback when the optional interactive dependency is unavailable.

**Files created/changed:**
- pygeoinf3D/pygeoinf3d/plot.py
- pygeoinf3D/tests/test_plot.py
- pygeoinf3D/pyproject.toml
- pygeoinf3D/docs/agent-docs/references/living/pygeoinf3d-reference.md
- pygeoinf3D/demos/least_norm_inversion_demo.ipynb
- pygeoinf3D/docs/agent-docs/active-plans/pygeoinf3d-interactive-slice-slider-plan.md

**Functions created/changed:**
- `plot_function3d_axis_slider`
- `_validate_n_slices`
- `_import_plotly_graph_objects`
- `_parse_axis` (extended to accept NumPy integer axes)

**Tests created/changed:**
- `test_plot_axis_slider_smoke`
- `test_plot_axis_slider_x_axis_labels`
- `test_plot_axis_slider_y_axis_labels`
- `test_plot_axis_slider_numpy_integer_axis`
- `test_plot_axis_slider_validation`
- `test_plot_axis_slider_requires_plotly`
- Verified focused plot suite: `12 passed in 2.15s`
- Verified full suite: `430 passed in 2.49s`
- Verified notebook execution: interactive slider renders in the least-norm demo notebook

**Review Status:** APPROVED with minor warnings

**Git Commit Message:**
feat(plot): add optional interactive slice slider for Function3D

- Add a Plotly-based axis slider viewer in pygeoinf3d.plot while keeping the static Matplotlib slice API unchanged
- Cover x/y/z traversal, NumPy integer axes, and missing-Plotly behavior with focused plot tests
- Document the interactive extra and use the new viewer in the least-norm demo notebook

Plan: pygeoinf3D/docs/agent-docs/active-plans/pygeoinf3d-interactive-slice-slider-plan.md
Phase: 1 of 1
Related: pygeoinf3D/docs/agent-docs/active-plans/pygeoinf3d-interactive-slice-slider-phase-1-complete.md
