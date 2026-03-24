## Plan: Resolving Target Kernel Viewer

Add a new Plotly helper in `pygeoinf3d.plot` that supports side-by-side panels with one shared slice slider and independent per-panel dropdowns. Then update the SOLA noise demo notebook/script to use a small hand-picked preview set so resolving and target kernels can be compared directly without paying the cost of building and plotting the full property set.

**Phases 3**
1. **Phase 1: Add Shared-Slider Multi-Panel Dropdown Viewer**
    - **Objective:** Implement a new plotting API that renders side-by-side panels with one shared slider and one dropdown per panel.
    - **Files/Functions to Modify/Create:** `pygeoinf3D/pygeoinf3d/plot.py`, `pygeoinf3D/tests/test_plot.py`
    - **Tests to Write:** smoke test for figure structure, shared-slider frame update test, per-panel dropdown isolation test, validation tests for empty panels, label-length mismatches, and bounding-box mismatches
    - **Steps:**
        1. Write failing tests for a new multi-panel dropdown plus shared-slider viewer.
        2. Implement the minimal plotting helper using the existing shared-slider and dropdown patterns.
        3. Run the focused plot tests and confirm the new helper behaves correctly.

2. **Phase 2: Update SOLA Demo To Use Small Preview Sets**
    - **Objective:** Replace the current interleaved single-dropdown kernel cell with a true side-by-side preview using a small hand-picked set of target indices.
    - **Files/Functions to Modify/Create:** `pygeoinf3D/demos/box/sola/sola_noise.ipynb`, `pygeoinf3D/demos/box/sola/sola_noise.py`
    - **Tests to Write:** execute the updated notebook cell and verify the script-side code path stays consistent with the new plotting API
    - **Steps:**
        1. Update the notebook cell to build only a small preview index set.
        2. Avoid wasteful identity-matrix allocation patterns in the demo code when constructing basis vectors.
        3. Switch the notebook and script to the new two-panel viewer and execute the notebook cell to validate the result.

3. **Phase 3: Update Reference Documentation**
    - **Objective:** Keep the living reference aligned with the new public plotting API and the revised demo usage.
    - **Files/Functions to Modify/Create:** `pygeoinf3D/docs/agent-docs/references/living/pygeoinf3d-reference.md`
    - **Tests to Write:** none beyond the focused plot tests and notebook execution already performed in earlier phases
    - **Steps:**
        1. Document the new viewer signature and behavior.
        2. Note the intended use case for preview comparisons in notebooks.
        3. Re-run the focused plot test file and confirm no regressions were introduced.

**Open Questions 2**
1. The new API will be general for any number of panels, while the notebook will use two panels specifically for resolving versus target comparison.
2. The notebook preview will use a small hand-picked subset of target indices rather than evenly spaced indices for easier scientific interpretation.