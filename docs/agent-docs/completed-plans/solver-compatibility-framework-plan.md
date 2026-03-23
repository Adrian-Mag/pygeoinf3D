## Plan: Solver Compatibility & Dispatcher Framework

Establish an **explicit compatibility framework** for pygeoinf3D solvers so that the
(geometry, operator type, functional type) requirements of each solver are machine-readable,
testable, and self-documenting. This prevents combinatorial explosion as new bespoke solvers
are added by enforcing a single `SolverRegistry` as the authoritative source of compatibilities,
backed by a lightweight `SolverRouter` that guides users to the right solver for their problem.

---

**Phases (4 total)**

1. **Phase 1: SolverSpec, SolverRegistry, and compatibility tests**
    - **Objective:** Create a `SolverSpec` dataclass that captures every solver's compatibility
      contract (required geometry class, required operator type, accepted functional types,
      assumed preconditions, performance tier, and human-readable notes). Populate a module-level
      `SOLVER_REGISTRY` with entries for `SpectralGramSolver`, `SOLAOperator3D`, and `KLSampler3D`.
    - **Files/Functions to Modify/Create:**
      - Create `pygeoinf3d/solvers/registry.py` — `SolverSpec` dataclass, `SolverRegistry` class,
        `SOLVER_REGISTRY` module-level singleton
      - Update `pygeoinf3d/solvers/__init__.py` to export `SolverSpec`, `SolverRegistry`, `SOLVER_REGISTRY`
      - Update `pygeoinf3d/__init__.py` to export `SolverRegistry`, `SOLVER_REGISTRY`
      - Create `tests/solvers/test_solver_registry.py`
    - **Tests to Write:**
      - `test_registry_has_three_entries` — registry has entries for all three known solvers
      - `test_spectral_gram_spec_fields` — verify SpectralGramSolver spec has correct geometry/operator/form types
      - `test_sola_spec_fields` — verify SOLAOperator3D spec
      - `test_kl_spec_fields` — verify KLSampler3D spec
      - `test_find_by_name` — registry lookup by solver name string
      - `test_list_all` — registry lists all registered solvers
    - **Steps:**
        1. Write failing tests in `test_solver_registry.py`
        2. Implement `SolverSpec` dataclass with fields: `name`, `solver_class`, `geometry_types`,
           `operator_types`, `functional_types`, `preconditions`, `performance_tier`, `notes`
        3. Implement `SolverRegistry` with `register()`, `find_by_name()`, `list_all()` methods
        4. Populate `SOLVER_REGISTRY` with entries for all three current solvers
        5. Export from `solvers/__init__.py` and top-level `__init__.py`
        6. Run tests to confirm all pass

2. **Phase 2: SolverRouter with decision-tree logic and tests**
    - **Objective:** Create a `SolverRouter` class that, given a space + list-of-forms,
      inspects the problem structure and recommends the best solver (from the registry),
      with reasons why and fallback suggestions when no fast path applies.
    - **Files/Functions to Modify/Create:**
      - Create `pygeoinf3d/solvers/router.py` — `SolverRouter` class with `recommend()` method
      - Update `pygeoinf3d/solvers/__init__.py` to export `SolverRouter`
      - Update `pygeoinf3d/__init__.py` to export `SolverRouter`
      - Create `tests/solvers/test_solver_router.py`
    - **Tests to Write:**
      - `test_spectral_gram_recommended_for_sobolev_path_forms` — canonical fast path
      - `test_sola_recommended_for_sobolev_volume_kernels` — volume kernel case
      - `test_kl_recommended_for_gaussian_sampling` — sampling case
      - `test_router_returns_recommendation_object` — result has `solver_spec`, `reason`, `warnings` fields
      - `test_router_warns_on_slow_path` — warns when only generic fallback applies
      - `test_router_no_recommendation_for_unsupported` — raises or returns None for unknown combination
    - **Steps:**
        1. Write failing tests in `test_solver_router.py`
        2. Define `SolverRecommendation` namedtuple: `solver_spec`, `reason`, `warnings`
        3. Implement `SolverRouter.recommend(space, forms)` with decision chain:
           - Check if Sobolev3D + spectral weight + all PathIntegralFunctional → SpectralGramSolver (fast)
           - Check if Sobolev3D or Lebesgue3D + any LinearForm → SOLAOperator3D (general)
           - Check if SpectralOperator3D + sampling intent → KLSampler3D
           - Fall through to generic pygeoinf warning
        4. Export from `solvers/__init__.py` and top-level `__init__.py`
        5. Run tests to confirm all pass

3. **Phase 3: Enhanced solver docstrings with explicit Requirements/Performance/Raises sections**
    - **Objective:** Rewrite solver class docstrings to contain explicit structured sections:
      `Requires:` (geometry, operator type, form type), `Performance:` (O() complexity, typical speedup),
      `Raises:` (what invalid inputs trigger what errors and why), `See Also:` (alternatives from registry).
      Also update runtime validation in each solver to produce actionable error messages naming alternatives.
    - **Files/Functions to Modify/Create:**
      - Update class docstring in `pygeoinf3d/solvers/spectral_gram.py`
      - Update class docstring in `pygeoinf3d/operators/sola.py` (SOLAOperator3D section)
      - Update class docstring in `pygeoinf3d/sampling/kl_sampler.py` (KLSampler3D section)
      - Update `SpectralGramSolver.__init__` TypeError messages to reference alternatives
    - **Tests to Write:**
      - `test_spectral_gram_wrong_space_type_error_message` — error string mentions SOLAOperator3D as alternative
      - `test_spectral_gram_wrong_operator_type_error_message` — error string mentions BesselSobolevBox
    - **Steps:**
        1. Write failing tests that assert error messages contain specific guidance text
        2. Rewrite `SpectralGramSolver` class docstring with Requirements/Performance/Raises/See Also sections
        3. Update `__init__` TypeErrors to include alternative solver suggestion in message
        4. Rewrite `SOLAOperator3D` class docstring with Requirements/Performance/See Also sections
        5. Rewrite `KLSampler3D` class docstring with Requirements/Performance/See Also sections
        6. Run tests to confirm all pass

4. **Phase 4: SOLVER_SELECTION_GUIDE and living reference update**
    - **Objective:** Create a user-facing `docs/SOLVER_SELECTION_GUIDE.md` with an ASCII decision
      tree, a compatibility matrix table, performance tiers, and usage examples. Update the
      living reference to include the registry overview and new exports.
    - **Files/Functions to Modify/Create:**
      - Create `pygeoinf3D/docs/SOLVER_SELECTION_GUIDE.md`
      - Update `docs/agent-docs/references/living/pygeoinf3d-reference.md` with solver registry section
    - **Tests to Write:**
      - None (documentation only)
    - **Steps:**
        1. Write `SOLVER_SELECTION_GUIDE.md` with: decision tree flowchart (ASCII), compatibility
           matrix (geometry × operator × form → solver), performance tiers, code examples for each
           recommended solver path, and entry point via `SolverRouter.recommend()`
        2. Update living reference: add "Solver Registry" sub-section listing all specs, new exports,
           and link to `SOLVER_SELECTION_GUIDE.md`

---

**Open Questions**
1. Should solvers self-register via a decorator at class-definition time, or use a centralized registry? (Chosen: centralized — avoids import-time side effects.)
2. `KLSampler3D` is not strictly an inversion solver — should it be in the same registry or a separate `SamplerRegistry`? (Chosen: same registry with `purpose="sampling"` tag, to keep one lookup point.)
3. What is the right interface for `SolverRouter.recommend()` — inspect live objects or accept type tokens? (Chosen: live objects — router can check `isinstance` and structural properties like `has_tensor_product_structure`.)
