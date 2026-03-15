## Plan: pygeoinf3D Synced Slice Comparison

Extend the optional Plotly slider plotting in `pygeoinf3d.plot` so multiple `Function3D` objects can be visualized side by side with one shared slider. Update the least-norm demo notebook to compare the true model, the least-norm solution, and the absolute total kernel sensitivity using the synced interactive viewer.

**Phases 1**
1. **Phase 1: Implement And Validate Synced Multi-Function Slice Viewer**
    - **Objective:** Add a synced multi-panel Plotly slider API, cover it with focused tests, document it, and use it in the least-norm demo notebook with absolute kernel sensitivity.
    - **Files/Functions to Modify/Create:** `pygeoinf3D/pygeoinf3d/plot.py`, `pygeoinf3D/tests/test_plot.py`, `pygeoinf3D/docs/agent-docs/references/living/pygeoinf3d-reference.md`, `pygeoinf3D/demos/least_norm_inversion_demo.ipynb`
    - **Tests to Write:** synced-frame structure tests, multi-panel labeling tests, validation tests for function lists and titles
    - **Steps:**
        1. Add a Plotly-based synced comparison viewer for multiple `Function3D` objects sharing one axis slider.
        2. Update the notebook to build the absolute total sensitivity function from the kernels and display true model, least-norm solution, and absolute total sensitivity in one synced figure.
        3. Run focused plot tests, the full suite, and the updated notebook cells to validate the feature.

**Open Questions 1**
1. Absolute sensitivity is defined here as `abs(sum_i k_i(x))`, where `k_i` are the volumetric kernel functions used by the SOLA operator.
