## Plan: pygeoinf3D Least-Norm Demo

Create a compact tutorial notebook in `pygeoinf3D/demos/` showing a noiseless minimum-norm inversion with a `pygeoinf3d.SOLAOperator3D` built from Gaussian volumetric kernels on a `BoxDomain`. The notebook will use pygeoinf’s existing `LinearForwardProblem` and `LinearMinimumNormInversion` workflow while keeping the model side basis-free.

**Phases 1**
1. **Phase 1: Create And Validate Least-Norm Demo Notebook**
    - **Objective:** Add a runnable notebook under `pygeoinf3D/demos/` that constructs a SOLA operator, generates noiseless synthetic data, computes the minimum-norm solution, and visualizes representative slices.
    - **Files/Functions to Modify/Create:** `pygeoinf3D/demos/least_norm_inversion_demo.ipynb`, `pygeoinf3D/docs/agent-docs/references/living/pygeoinf3d-reference.md`
    - **Tests to Write:** notebook execution validation only
    - **Steps:**
        1. Create the notebook with markdown narrative and code cells for setup, forward model construction, inversion, and plotting.
        2. Run the notebook cells to confirm the inversion works and the plots render without errors.
        3. Update the living reference to mention the new demo notebook.

**Open Questions 1**
1. The notebook will live in `pygeoinf3D/demos/` and import pygeoinf as the inversion engine. This matches the approved scope.
