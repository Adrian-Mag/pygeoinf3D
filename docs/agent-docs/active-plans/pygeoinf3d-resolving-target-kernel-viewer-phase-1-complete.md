## Phase 1 Complete: Add Shared-Slider Multi-Panel Dropdown Viewer

Implemented a new Plotly helper for side-by-side panels with one shared slice
slider and one independent dropdown per panel. The phase also added focused
regression tests, tightened API validation, updated public exports, and kept
the living reference aligned with the new plotting entry point.

**Files created/changed:**
- pygeoinf3D/pygeoinf3d/plot.py
- pygeoinf3D/tests/test_plot.py
- pygeoinf3D/pygeoinf3d/__init__.py
- pygeoinf3D/docs/agent-docs/references/living/pygeoinf3d-reference.md

**Functions created/changed:**
- compare_panels_axis_dropdown_sliders
- _import_plotly_make_subplots

**Tests created/changed:**
- test_compare_panels_axis_dropdown_sliders_smoke
- test_compare_panels_axis_dropdown_sliders_frames
- test_compare_panels_axis_dropdown_sliders_dropdown_isolation
- test_compare_panels_axis_dropdown_sliders_per_panel_color
- test_compare_panels_axis_dropdown_sliders_global_vmin_vmax
- test_compare_panels_axis_dropdown_sliders_custom_labels
- test_compare_panels_axis_dropdown_sliders_single_panel
- test_compare_panels_axis_dropdown_sliders_validation_empty_outer
- test_compare_panels_axis_dropdown_sliders_validation_empty_panel
- test_compare_panels_axis_dropdown_sliders_labels_outer_mismatch
- test_compare_panels_axis_dropdown_sliders_labels_inner_mismatch
- test_compare_panels_axis_dropdown_sliders_validation_bbox_mismatch
- test_compare_panels_axis_dropdown_sliders_validation_panel_titles_mismatch
- test_compare_panels_axis_dropdown_sliders_importerror_when_plotly_missing

**Review Status:** APPROVED

**Git Commit Message:**
feat(plot): add shared-slider multi-panel kernel viewer

- Add compare_panels_axis_dropdown_sliders for synced slice comparison
- Cover dropdown isolation, frame updates, and validation edge cases
- Export the helper and document the new viewer in the living reference

Plan: pygeoinf3D/docs/agent-docs/active-plans/pygeoinf3d-resolving-target-kernel-viewer-plan.md
Phase: 1 of 3
Related: pygeoinf3D/docs/agent-docs/active-plans/pygeoinf3d-resolving-target-kernel-viewer-phase-1-complete.md