## Phase 1 Complete: Extract `_build_L_matrix` from `SpectralGramSolver`

Extracted L-matrix assembly from `SpectralGramSolver.__init__` into a reusable `@staticmethod _build_L_matrix(sobolev_space, forms, *, resolution=20)` returning `(L: ndarray[m,N], nu: ndarray[N,])`. `__init__` now delegates to it. Supports mixed PathIntegralFunctional + VolumeKernel3D stacks and raises `TypeError` loudly for unsupported form types or non-spectral inverse weights.

**Files created/changed:**
- `pygeoinf3D/pygeoinf3d/solvers/spectral_gram.py`
- `pygeoinf3D/tests/operators/test_cross_gram.py` (new file)

**Functions created/changed:**
- `SpectralGramSolver._build_L_matrix` (new `@staticmethod`)
- `SpectralGramSolver.__init__` (refactored to delegate to `_build_L_matrix`)

**Tests created/changed:**
- `TestBuildLMatrix::test_path_forms_matches_interaction_matrix`
- `TestBuildLMatrix::test_vol_forms_matches_interaction_matrix`
- `TestBuildLMatrix::test_mixed_forms_matches_interaction_matrix`
- `TestBuildLMatrix::test_rejects_non_spectral_sobolev`

**Review Status:** APPROVED

**Git Commit Message:**
```
refactor(spectral_gram): extract _build_L_matrix static helper

- Add @staticmethod _build_L_matrix(sobolev_space, forms, resolution) -> (L, nu)
- Refactor __init__ to delegate L-matrix assembly to _build_L_matrix
- Supports mixed PathIntegralFunctional + VolumeKernel3D form stacks
- Raises TypeError loudly for non-spectral inverse_weight

Plan: pygeoinf3D/docs/agent-docs/active-plans/cross-gram-sola-fast-path-plan.md
Phase: 1 of 3
Related: pygeoinf3D/docs/agent-docs/active-plans/cross-gram-sola-fast-path-phase-1-complete.md
```
