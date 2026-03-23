## Plan Complete: Solver Compatibility & Dispatcher Framework

An explicit compatibility framework was built for pygeoinf3D solvers, making (geometry, operator, functional) requirements machine-readable, testable, and self-documenting. The `SolverRegistry` is now the authoritative source of solver compatibility, backed by a `SolverRouter` that guides users to the right solver for their problem via `router.recommend(space, forms)`.

**Phases Completed:** 4 of 4
1. ✅ Phase 1: SolverSpec, SolverRegistry, and compatibility tests
2. ✅ Phase 2: SolverRouter with decision-tree logic and tests
3. ✅ Phase 3: Enhanced solver docstrings and validation error messages
4. ✅ Phase 4: SOLVER_SELECTION_GUIDE.md and living reference update

**All Files Created/Modified:**
- `pygeoinf3d/solvers/registry.py` (new) — `SolverSpec` dataclass, `SolverRegistry` class, `SOLVER_REGISTRY` singleton
- `pygeoinf3d/solvers/router.py` (new) — `SolverRouter`, `SolverRecommendation` with 3-level decision logic
- `pygeoinf3d/solvers/__init__.py` (modified) — exports `SpectralGramSolver`, `SolverSpec`, `SolverRegistry`, `SOLVER_REGISTRY`, `SolverRouter`, `SolverRecommendation`
- `pygeoinf3d/__init__.py` (modified) — top-level exports for all new registry/router symbols
- `pygeoinf3d/providers/tensor_product.py` (modified) — added `has_tensor_product_structure = True` to `BoxLaplacianSpectrumProvider`
- `pygeoinf3d/solvers/spectral_gram.py` (modified) — structured docstring, improved error messages with alternative suggestions, + n_quad consistency validation
- `pygeoinf3d/operators/sola.py` (modified) — structured class docstring with Requires/Performance/See Also
- `pygeoinf3d/sampling/kl_sampler.py` (modified) — structured class docstring with Requires/Performance/See Also
- `docs/SOLVER_SELECTION_GUIDE.md` (new) — ASCII decision tree, compatibility matrix, "adding a new solver" guide
- `docs/agent-docs/references/living/pygeoinf3d-reference.md` (modified) — new Solver Registry & Router section, updated status
- `tests/solvers/test_solver_registry.py` (new) — 15 tests for registry lookups
- `tests/solvers/test_solver_router.py` (new) — 12 tests for router decision paths
- `tests/solvers/test_solver_validation.py` (new) — 3 tests for improved error messages

**Key Functions/Classes Added:**
- `SolverSpec` — dataclass capturing every solver's compatibility contract
- `SolverRegistry` — (register, find_by_name, list_all, find_by_purpose)
- `SOLVER_REGISTRY` — module-level singleton populated with SpectralGramSolver, SOLAOperator3D, KLSampler3D specs
- `SolverRecommendation` — dataclass with solver_spec, reason, warnings fields
- `SolverRouter` — (recommend, recommend_sampler, list_available)

**Test Coverage:**
- Total tests written: 30 (15 registry + 12 router + 3 validation)
- All tests passing: ✅
- Total suite: **861 passed, 0 failures**

**Recommendations for Next Steps:**
- Add `SolverRouter.recommend()` calls to demo notebooks to illustrate the workflow
- When a new bespoke solver is added, follow the 5-step guide in `docs/SOLVER_SELECTION_GUIDE.md`
- Consider adding a `recommend_for_inversion(space, forms, data_covariance=None)` convenience wrapper that returns the instantiated solver rather than a spec
