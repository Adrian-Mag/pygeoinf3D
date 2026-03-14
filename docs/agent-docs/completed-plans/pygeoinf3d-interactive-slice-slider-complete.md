## Plan Complete: pygeoinf3D Interactive Slice Slider

Implemented an optional Plotly-based interactive slice viewer for `Function3D` objects on box-like 3D domains, without changing the existing lightweight Matplotlib slice API. The feature now supports interactive traversal along `x`, `y`, or `z`, is packaged behind an `interactive` extra, is covered by focused tests, and is demonstrated in the least-norm inversion notebook with a graceful fallback when Plotly is unavailable.

**Phases Completed:** 1 of 1
1. ✅ Phase 1: Implement And Validate Interactive Slice Viewer

**All Files Created/Modified:**
- pygeoinf3D/pygeoinf3d/plot.py
- pygeoinf3D/tests/test_plot.py
- pygeoinf3D/pyproject.toml
- pygeoinf3D/docs/agent-docs/references/living/pygeoinf3d-reference.md
- pygeoinf3D/demos/least_norm_inversion_demo.ipynb
- pygeoinf3D/docs/agent-docs/active-plans/pygeoinf3d-interactive-slice-slider-plan.md
- pygeoinf3D/docs/agent-docs/active-plans/pygeoinf3d-interactive-slice-slider-phase-1-complete.md

**Key Functions/Classes Added:**
- `plot_function3d_axis_slider`
- `_validate_n_slices`
- `_import_plotly_graph_objects`
- `_parse_axis` (NumPy integer support)

**Test Coverage:**
- Total tests added/updated in `tests/test_plot.py`: 6
- Focused plot tests passing: ✅ `12 passed in 2.15s`
- Full suite passing: ✅ `430 passed in 2.49s`
- Demo notebook executed successfully: ✅

**Recommendations for Next Steps:**
- Consider adding synchronized multi-panel slider viewers (for true / recovered / difference) if interactive comparison becomes a common workflow.
- Consider a lightweight helper to choose between static Matplotlib output and Plotly slider output from a single public entry point if the plotting surface grows further.
