## Phase 1 Complete: Implement And Validate Synced Multi-Function Slice Viewer

Added a synced multi-panel Plotly slice viewer for `Function3D` comparisons, updated the least-norm demo to compare the true model, least-norm solution, and absolute total sensitivity, and validated the package with focused and full test runs. The demo notebook now executes cleanly with the new interactive comparison after a kernel restart.

**Files created/changed:**
- pygeoinf3D/pygeoinf3d/plot.py
- pygeoinf3D/tests/test_plot.py
- pygeoinf3D/demos/least_norm_inversion_demo.ipynb
- pygeoinf3D/docs/agent-docs/references/living/pygeoinf3d-reference.md
- pygeoinf3D/docs/agent-docs/active-plans/pygeoinf3d-synced-slice-comparison-plan.md

**Functions created/changed:**
- `_import_plotly_graph_objects`
- `_import_plotly_make_subplots`
- `compare_function3d_axis_sliders`

**Tests created/changed:**
- `test_compare_function3d_axis_sliders_smoke`
- `test_compare_function3d_axis_sliders_titles`
- `test_compare_function3d_axis_sliders_single_panel`
- `test_compare_function3d_axis_sliders_axis_labels`
- `test_compare_function3d_axis_sliders_shared_color_limits`
- `test_compare_function3d_axis_sliders_validation_empty`
- `test_compare_function3d_axis_sliders_validation_titles_length`
- `test_compare_function3d_axis_sliders_validation_bbox_mismatch`
- `test_compare_function3d_axis_sliders_requires_plotly`

**Review Status:** APPROVED

**Git Commit Message:**
feat(plot): add synced Function3D slice comparison

- add a shared-slider Plotly comparison viewer for multiple Function3D panels
- update the least-norm demo to compare true model, solution, and absolute sensitivity
- extend plot tests and living reference for the new interactive comparison API

Plan: pygeoinf3D/docs/agent-docs/active-plans/pygeoinf3d-synced-slice-comparison-plan.md
Phase: 1 of 1
Related: pygeoinf3D/docs/agent-docs/active-plans/pygeoinf3d-synced-slice-comparison-phase-1-complete.md
