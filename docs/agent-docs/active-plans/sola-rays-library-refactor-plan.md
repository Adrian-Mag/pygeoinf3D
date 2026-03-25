## Plan: SOLA Rays Library Refactor

Refactor `sola_rays.ipynb` so it uses `pygeoinf` and `pygeoinf3d` abstractions wherever those packages already provide a mathematically equivalent operation. The main replacements are operator adjoints and solver classes for least-norm and representer-based constructions; the remaining custom notebook logic is unimodularity and grid-to-field interpolation, because no reusable library API was found for those yet.

**Phases 4**
1. **Phase 1: Replace Custom Operator Algebra**
    - **Objective:** Remove notebook-level constructions that can be expressed directly through `SOLAOperator3D`, adjoints, and existing operator APIs.
    - **Files/Functions to Modify/Create:** `/home/adrian/PhD/Inferences/pygeoinf3D/demos/box/sola/sola_rays.ipynb`
    - **Tests to Write:** Execute the edited total-sensitivity and resolving-kernel cells and confirm they still produce `Function3D` outputs and plots.
    - **Steps:**
        1. Edit the total-sensitivity cell so it uses `G.adjoint(np.ones(N_d))` instead of a custom `Function3D` summation loop.
        2. Edit the resolving-kernel cell so each resolving kernel is built with `G.adjoint(X_uni[k, :])` instead of a manual weighted representer sum.
        3. Remove any helper code made redundant by these replacements.
        4. Run the affected notebook cells to confirm they execute.

2. **Phase 2: Replace Manual Least-Norm Solve**
    - **Objective:** Use the dedicated `pygeoinf3d` solver abstraction for the Sobolev/path least-norm solution instead of manually solving and applying the adjoint.
    - **Files/Functions to Modify/Create:** `/home/adrian/PhD/Inferences/pygeoinf3D/demos/box/sola/sola_rays.ipynb`
    - **Tests to Write:** Execute the least-norm cell and confirm the solver returns a `Function3D` and a sensible residual.
    - **Steps:**
        1. Replace the manual least-norm block with `SpectralGramSolver`.
        2. Pass the notebook noise covariance through the solver in the same mathematical form as the current workflow.
        3. Run the least-norm and downstream comparison cells.

3. **Phase 3: Tighten Remaining Library Usage**
    - **Objective:** Replace any remaining ad hoc notebook code where a concrete library abstraction exists, while preserving the custom geometry plot and unsupported custom math.
    - **Files/Functions to Modify/Create:** `/home/adrian/PhD/Inferences/pygeoinf3D/demos/box/sola/sola_rays.ipynb`
    - **Tests to Write:** Execute the affected cells after cleanup.
    - **Steps:**
        1. Audit imports and remove now-unused helper code introduced only for superseded notebook logic.
        2. Preserve the custom Plotly geometry preview.
        3. Keep `_make_prop_fn` and the unimodularity correction custom unless a concrete existing API is identified during implementation.
        4. Run the affected inference and visualization cells.

4. **Phase 4: Notebook Validation and Cleanup**
    - **Objective:** Ensure the notebook executes coherently after the refactor and that only irreducible custom logic remains.
    - **Files/Functions to Modify/Create:** `/home/adrian/PhD/Inferences/pygeoinf3D/demos/box/sola/sola_rays.ipynb`
    - **Tests to Write:** Execute the edited notebook path from imports through least-norm solution.
    - **Steps:**
        1. Run the edited cells in order.
        2. Fix any symbol, typing, or operator-shape regressions introduced by the refactor.
        3. Leave a clear boundary in the notebook between library-backed sections and the few remaining custom formulas.

**Open Questions 3**
1. Keep the manual SOLA matrix construction itself for pedagogical transparency, or replace it if a fully equivalent existing abstraction is found later?
2. Are any downstream script companions to be kept in sync now? Current answer: no, notebook only.
3. Should unsupported custom math be promoted into library APIs later? Current answer: no, keep notebook-scoped for now.
