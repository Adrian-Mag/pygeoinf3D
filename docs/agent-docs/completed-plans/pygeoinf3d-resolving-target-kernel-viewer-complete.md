## Plan Complete: Resolving Target Kernel Viewer

Added a new Plotly plotting helper for side-by-side panel comparisons with
independent per-panel dropdowns and one shared slice slider, then updated the
SOLA noise demo notebook and script to use a small 5-point diagonal preview
of resolving vs target kernels through that new helper. The viewer is now
fast to run (only 5 resolving kernels) and immediately shows how localised
each resolving kernel is relative to its target.

**Phases Completed:** 3 of 3
1. ✅ Phase 1: Add Shared-Slider Multi-Panel Dropdown Viewer
2. ✅ Phase 2: Update SOLA Demo To Use Small Preview Sets
3. ✅ Phase 3: Update Reference Documentation (completed inline during Phase 1 and 2)

**All Files Created/Modified:**
- pygeoinf3D/pygeoinf3d/plot.py
- pygeoinf3D/tests/test_plot.py
- pygeoinf3D/pygeoinf3d/__init__.py
- pygeoinf3D/demos/box/sola/sola_noise.ipynb
- pygeoinf3D/demos/box/sola/sola_noise.py
- pygeoinf3D/docs/agent-docs/references/living/pygeoinf3d-reference.md
- pygeoinf3D/docs/agent-docs/active-plans/pygeoinf3d-resolving-target-kernel-viewer-plan.md
- pygeoinf3D/docs/agent-docs/active-plans/pygeoinf3d-resolving-target-kernel-viewer-phase-1-complete.md
- pygeoinf3D/docs/agent-docs/active-plans/pygeoinf3d-resolving-target-kernel-viewer-phase-2-complete.md

**Key Functions/Classes Added:**
- compare_panels_axis_dropdown_sliders

**Test Coverage:**
- Total tests written: 14 focused tests for the new helper
- All tests verified in the workspace Python 3.11 environment
- Notebook cell executed successfully (exit code 0, Plotly figure produced)

**Recommendations for Next Steps:**
- PREVIEW_IDX could be made a notebook parameter for exploratory use
- Per-panel vmin/vmax override option could be added to the helper for
  asymmetric kernel comparisons
