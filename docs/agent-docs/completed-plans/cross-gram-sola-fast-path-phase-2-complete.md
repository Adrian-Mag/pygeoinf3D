## Phase 2 Complete: Cross-Gram `__matmul__` Fast Path

Added cross-gram fast path to `SOLAOperator3D.__matmul__` so that `T @ G.adjoint` (and `G @ T.adjoint`) between two different `SOLAOperator3D` instances on the same `Sobolev3D` domain returns a dense `MatrixLinearOperator` computed via $C_{TG} = L_T \operatorname{diag}(\nu) L_G^\top$. Caches by `id(other_sola)`. Fails loudly if forms are unsupported. Reduced `p_cov.matrix()` from ~160 s to 0.214 s (750× speedup).

**Files created/changed:**
- `pygeoinf3D/pygeoinf3d/operators/sola.py`
- `pygeoinf3D/tests/operators/test_cross_gram.py`

**Functions created/changed:**
- `SOLAOperator3D.__init__` (added `_cross_gram_cache` dict)
- `SOLAOperator3D._try_build_fast_cross_gram_operator` (new method)
- `SOLAOperator3D.__matmul__` (extended with cross-gram `elif` branch)

**Tests created/changed:**
- `TestCrossGramFastPath::test_T_at_G_adjoint_returns_matrix_linear_operator`
- `TestCrossGramFastPath::test_G_at_T_adjoint_returns_matrix_linear_operator`
- `TestCrossGramFastPath::test_cross_gram_value_matches_slow_path`
- `TestCrossGramFastPath::test_self_gram_still_uses_existing_fast_path`
- `TestCrossGramFastPath::test_cross_gram_is_cached`
- `TestCrossGramFastPath::test_lebesgue_domain_skips_fast_path`

**Review Status:** APPROVED (with minor revision: added unsupported-form TypeError check handled in Phase 3)

**Git Commit Message:**
```
feat(sola): cross-gram __matmul__ fast path for T @ G.adjoint

- Add _try_build_fast_cross_gram_operator using spectral C_TG = L_T diag(nu) L_G.T
- Extend __matmul__ with elif branch detecting two SOLAOperator3D on same Sobolev3D
- Cache result by id(other_sola) to avoid recomputation
- Fail loudly (TypeError) for unsupported form types
- 6 new tests; 1011 total passing

Plan: pygeoinf3D/docs/agent-docs/active-plans/cross-gram-sola-fast-path-plan.md
Phase: 2 of 3
Related: pygeoinf3D/docs/agent-docs/active-plans/cross-gram-sola-fast-path-phase-2-complete.md
```
