## Phase 3 Complete: Refactor Existing Wrappers Onto the Generalized Engine

Phase 3 eliminated duplicated quadrature/projection logic by making `BoxLaplacian` a direct subclass of `GeneralizedBoxLaplacian` and routing the `InverseBoxLaplacian` pseudoinverse path through a `GeneralizedBoxLaplacian(power=-1)` delegate. All public APIs remain unchanged. The shift and strict nullspace modes in `InverseBoxLaplacian` retain their independent implementations because they are not representable by the generalized engine.

**Files created/changed:**
- `pygeoinf3D/pygeoinf3d/operators/laplacian_box.py`
- `pygeoinf3D/pygeoinf3d/operators/inverse_laplacian_box.py`
- `pygeoinf3D/tests/operators/test_box_laplacian.py`
- `pygeoinf3D/tests/operators/test_inverse_box_laplacian.py`
- `pygeoinf3D/docs/agent-docs/references/living/pygeoinf3d-reference.md`

**Functions created/changed:**
- `BoxLaplacian.__init__` (now delegates to `super().__init__(power=1, alpha=1.0)`)
- `BoxLaplacian.__repr__` (overridden to preserve existing format)
- `InverseBoxLaplacian.__init__` (constructs `_delegate` for pseudoinverse mode)
- `InverseBoxLaplacian._apply` (routes through `_delegate` when set)
- `InverseBoxLaplacian.get_eigenvalue` (routes through `_delegate` when set)
- `InverseBoxLaplacian.get_eigenfunction` (routes through `_delegate` when set)
- `InverseBoxLaplacian.n_modes` (routes through `_delegate` when set)

**Tests created/changed:**
- `TestBoxLaplacianWrapperEquivalence` (4 tests: isinstance check, `power==1`, `alpha==1.0`, apply match, repr format)
- `TestInverseBoxLaplacianWrapperEquivalence` (6 tests: delegate isinstance, apply match, `_delegate is None` for shift/strict, shift mode still functional)

**Review Status:** APPROVED (after removing unused `LinearOperator` import)

**Git Commit Message:**
refactor(operators): make BoxLaplacian and InverseBoxLaplacian delegate to generalized engine

- BoxLaplacian is now a thin subclass of GeneralizedBoxLaplacian (power=1, alpha=1.0)
- InverseBoxLaplacian pseudoinverse mode delegates to GeneralizedBoxLaplacian(power=-1)
- shift and strict modes retain independent implementation
- 10 new wrapper-equivalence tests, 612 total passing

Plan: pygeoinf3D/docs/agent-docs/active-plans/pygeoinf3d-generalized-box-laplacian-plan.md
Phase: 3 of 5
Related: pygeoinf3D/docs/agent-docs/active-plans/pygeoinf3d-generalized-box-laplacian-phase-3-complete.md
