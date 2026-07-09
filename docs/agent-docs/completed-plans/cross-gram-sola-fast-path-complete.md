## Plan Complete: Cross-Gram Fast Path for `T @ G.adjoint`

Implemented spectral cross-gram materialization so that `T @ G.adjoint` between two `SOLAOperator3D` instances on the same `Sobolev3D` domain returns a dense `MatrixLinearOperator` in ~0.2 ms instead of triggering the full lazy composition chain. This collapsed `p_cov.matrix()` from ~160 s to 0.214 s (750× speedup) with zero numerical error (`||p_mean_fast − p_mean_old|| / ||p_mean_old|| = 0.00e+00`).

**Phases Completed:** 3 of 3
1. ✅ Phase 1: Extract `_build_L_matrix` static helper from `SpectralGramSolver`
2. ✅ Phase 2: Cross-gram `__matmul__` fast path in `SOLAOperator3D`
3. ✅ Phase 3: Notebook verification & code review revisions

**All Files Created/Modified:**
- `pygeoinf3D/pygeoinf3d/solvers/spectral_gram.py`
- `pygeoinf3D/pygeoinf3d/operators/sola.py`
- `pygeoinf3D/tests/operators/test_cross_gram.py` (new)
- `pygeoinf3D/demos/box/sola/sola_rays.ipynb`
- `pygeoinf3D/docs/agent-docs/references/living/pygeoinf3d-reference.md`

**Key Functions/Classes Added:**
- `SpectralGramSolver._build_L_matrix(sobolev_space, forms, *, resolution=20) -> (L, nu)` — reusable spectral interaction-matrix builder
- `SOLAOperator3D._try_build_fast_cross_gram_operator(other_sola) -> MatrixLinearOperator` — computes $C_{TG} = L_T \operatorname{diag}(\nu) L_G^\top$
- `SOLAOperator3D._cross_gram_cache: dict[int, MatrixLinearOperator]` — per-instance cache keyed by `id(other_sola)`
- Extended `SOLAOperator3D.__matmul__` with cross-gram `elif` branch

**Test Coverage:**
- Total tests written: 11 (in `test_cross_gram.py`)
- All tests passing: ✅ (1012 total suite)

**Recommendations for Next Steps:**
- The residual 0.214 s in `p_cov.matrix()` is now dominated by pygeoinf column-by-column framework overhead; a batched `matrix()` implementation could reduce it further.
- The cache keys by `id(other_sola)` — consider `WeakKeyDictionary` if operators are short-lived in long-running sessions to avoid memory leaks.
