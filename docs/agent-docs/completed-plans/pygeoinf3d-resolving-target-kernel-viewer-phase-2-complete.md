## Phase 2 Complete: Update SOLA Demo To Use Small Preview Sets

The SOLA noise demo notebook and its companion script were updated to use the
new side-by-side viewer. The notebook's resolving-kernel cell now builds only
five diagonal preview targets, avoids allocating an N_p×N_p identity matrix,
and calls `compare_panels_axis_dropdown_sliders` to display resolving and
target kernels side-by-side with a shared slice slider. The updated cell
executed successfully and produced the expected interactive Plotly figure.

**Files created/changed:**
- pygeoinf3D/demos/box/sola/sola_noise.ipynb
- pygeoinf3D/demos/box/sola/sola_noise.py
- pygeoinf3D/docs/agent-docs/references/living/pygeoinf3d-reference.md

**Functions created/changed:**
- sola_noise.ipynb cell 20: replaced interleaved single-dropdown with
  compare_panels_axis_dropdown_sliders two-panel viewer
- sola_noise.py resolving-kernels section: same replacement + _e() helper

**Tests created/changed:**
- Notebook execution (execution count 30, success, Plotly figure produced)

**Review Status:** APPROVED

**Git Commit Message:**
feat(demos): update SOLA resolving-kernel viewer to two-panel UI

- Replace interleaved single-dropdown with compare_panels_axis_dropdown_sliders
- Use 5 diagonal preview indices; avoid np.eye(N_p) identity allocation
- Mirror changes in sola_noise.py companion script
- Register sola_noise demo in living reference file tree

Plan: pygeoinf3D/docs/agent-docs/active-plans/pygeoinf3d-resolving-target-kernel-viewer-plan.md
Phase: 2 of 3
Related: pygeoinf3D/docs/agent-docs/active-plans/pygeoinf3d-resolving-target-kernel-viewer-phase-2-complete.md
