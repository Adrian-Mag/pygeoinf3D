## Plan: Bessel Sobolev Demo Notebook

Create a new pygeoinf3D demo notebook showing `BesselSobolevBox` in both operator and covariance roles. The notebook will follow the existing demo style, support `VISUALIZATION_MODE = "static" | "interactive" | "both"`, and demonstrate positive orders as Sobolev-ladder operators and negative orders as Gaussian-measure covariance operators via `KLSampler3D`.

**Phases 3**
1. **Phase 1: Add Notebook Scaffold and Positive-Order Demo**
    - **Objective:** Create a new notebook in `pygeoinf3D/demos/` with imports, setup, visualization-mode controls, and positive-order operator examples.
    - **Files/Functions to Modify/Create:** `pygeoinf3D/demos/bessel_sobolev_3d_demo.ipynb`
    - **Tests to Write:** notebook sanity only; no new pytest tests
    - **Steps:**
        1. Write notebook markdown and setup cells following the existing demo conventions.
        2. Add box/space/provider/operator setup with Neumann boundary conditions and positive shift.
        3. Add positive-order sections showing spectral weights and action on a smooth test function.
        4. Include static and interactive visualization paths controlled by `VISUALIZATION_MODE`.
2. **Phase 2: Add Negative-Order Covariance and Sampling Demo**
    - **Objective:** Show negative-order `BesselSobolevBox` operators as covariance operators and visualize Gaussian samples drawn with `KLSampler3D`.
    - **Files/Functions to Modify/Create:** `pygeoinf3D/demos/bessel_sobolev_3d_demo.ipynb`
    - **Tests to Write:** notebook sanity only; no new pytest tests
    - **Steps:**
        1. Add sections explaining why negative order gives decaying covariance eigenvalues.
        2. Construct `KLSampler3D` instances from negative-order operators.
        3. Add sample galleries comparing smoothness for several negative orders.
        4. Add both static and interactive visualizations matching the other notebooks.
3. **Phase 3: Validate and Sync References**
    - **Objective:** Sanity-check the notebook code against the current API and update the living reference to include the new demo.
    - **Files/Functions to Modify/Create:** `pygeoinf3D/demos/bessel_sobolev_3d_demo.ipynb`, `pygeoinf3D/docs/agent-docs/references/living/pygeoinf3d-reference.md`
    - **Tests to Write:** run a lightweight Python validation snippet for imports/construction
    - **Steps:**
        1. Validate the notebook’s core imports and constructions in the configured Python environment.
        2. Add the notebook to the living reference demo inventory and repository layout comments.
        3. Confirm the notebook JSON is well-formed.

**Open Questions 1**
1. Visualization mode support should mirror the other notebooks: `static`, `interactive`, or `both`.
