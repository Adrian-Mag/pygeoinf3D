## Plan: Bessel Sobolev Box Operator

Implement a standalone box-domain Bessel-Sobolev spectral operator in pygeoinf3D, independent of any Sobolev-space wrapper. The operator will mirror the existing generalized box Laplacian structure, but use shifted spectral weights $\mu_j = (\mathrm{shift} + \lambda_j)^{\mathrm{order}/2}$ so it can later serve as a ladder operator between Sobolev spaces.

**Phases 3**
1. **Phase 1: Add Core Bessel-Sobolev Box Operator**
    - **Objective:** Introduce `BesselSobolevBox` as a standalone spectral operator on `Lebesgue3D`.
    - **Files/Functions to Modify/Create:** `pygeoinf3D/pygeoinf3d/operators/bessel_sobolev_box.py`, `pygeoinf3D/pygeoinf3d/operators/__init__.py`, `pygeoinf3D/tests/operators/test_bessel_sobolev_box.py`
    - **Tests to Write:** constructor validation, eigenvalue formula, identity case `order = 0`, positive and negative order checks
    - **Steps:**
        1. Write failing tests for constructor validation and spectral-weight semantics.
        2. Implement `BesselSobolevBox` as a subclass of the existing spectral operator base.
        3. Use the same box quadrature `_apply` pattern as the generalized box Laplacian.
        4. Export the class from the package root.
        5. Run focused tests.
2. **Phase 2: Verify Operator Action and Ladder Semantics**
    - **Objective:** Confirm the operator acts correctly on eigenfunctions and behaves consistently across positive, zero, and negative orders.
    - **Files/Functions to Modify/Create:** `pygeoinf3D/tests/operators/test_bessel_sobolev_box.py`
    - **Tests to Write:** apply-to-eigenfunction tests, inverse/negative-order behavior, self-adjointness via spectral symmetry, output-space consistency
    - **Steps:**
        1. Write failing tests for action on provider eigenfunctions.
        2. Verify `order = 0` gives the identity operator.
        3. Verify `order > 0` and `order < 0` use the same unified formula with exponent `order / 2`.
        4. Re-run focused tests.
3. **Phase 3: Document and Stabilize**
    - **Objective:** Document the operator and make its role explicit as an operator-level building block for later Sobolev spaces.
    - **Files/Functions to Modify/Create:** `pygeoinf3D/docs/agent-docs/references/living/pygeoinf3d-reference.md`
    - **Tests to Write:** only reruns unless API wording needs correction
    - **Steps:**
        1. Add the operator to the living reference with its spectral definition.
        2. Note that future Sobolev spaces should wrap this operator rather than embedding the logic themselves.
        3. Re-run the focused test file and archive the plan when complete.

**Open Questions 1**
1. Should `shift = 0` be allowed for Neumann providers, or should the first implementation require `shift > 0` to avoid the zero-mode singular case for negative orders?
