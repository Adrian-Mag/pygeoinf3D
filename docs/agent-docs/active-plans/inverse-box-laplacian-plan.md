## Plan: InverseBoxLaplacian

Implement `InverseBoxLaplacian` — the spectral (pseudo-)inverse of $-\Delta$ on a `BoxDomain` — as a `LinearOperator: Lebesgue3D → Lebesgue3D`. It mirrors `InverseLaplacian` in intervalinf but is entirely 3D-spectral, using the same quadrature-projection pattern as `BoxLaplacian`. Three strategies let users control the Neumann zero mode: skip it (pseudoinverse), shift $(λ_k + σ)^{-1}$, or refuse to construct. A user-configurable `eigenvalue_tol` (default `1e-10`) marks near-zero eigenvalues. No `alpha` scaling for now (consistent with `BoxLaplacian`).

**Phases (2)**

1. **Phase 1: TDD implementation**
    - **Objective:** Implement `InverseBoxLaplacian` following strict TDD — failing tests first, then minimal code to pass.
    - **Files/Functions to Modify/Create:**
        - Create `pygeoinf3D/pygeoinf3d/operators/inverse_laplacian_box.py` — new `InverseBoxLaplacian(LinearOperator)` class
        - Create `pygeoinf3D/tests/operators/test_inverse_box_laplacian.py` — full test suite
    - **Tests to Write:**
        - `test_construct_dirichlet` — constructs without error on Dirichlet (no zero modes)
        - `test_construct_neumann_pseudoinverse` — constructs on Neumann with `null_space_handling='pseudoinverse'`
        - `test_construct_shift` — constructs with `null_space_handling='shift', shift=0.1`
        - `test_construct_strict_neumann_raises` — Neumann + `null_space_handling='strict'` raises `ValueError` at construction
        - `test_is_linear_operator` — instance of `LinearOperator`
        - `test_domain_codomain_lebesgue3d` — both domain and codomain are the space
        - `test_eigenfunction_mapped_to_inverse_eigenvalue_dirichlet` — applying to Φ_k yields (1/λ_k)Φ_k pointwise (Dirichlet, several modes)
        - `test_eigenfunction_mapped_to_inverse_eigenvalue_neumann_pseudoinverse` — same for non-zero Neumann modes
        - `test_zero_mode_suppressed_neumann` — Neumann zero mode (λ_0 = 0) maps to zero output
        - `test_round_trip_dirichlet` — `InverseBoxLaplacian(BoxLaplacian(f))` ≈ projection of `f` onto spectral basis
        - `test_round_trip_neumann_pseudoinverse` — same ignoring the zero-mode component
        - `test_shift_mode_eigenvalue_scaling` — applying to Φ_k yields (1/(λ_k + σ))Φ_k
        - `test_self_adjoint` — smoke test: ⟨Af, g⟩ ≈ ⟨f, Ag⟩
    - **Steps:**
        1. Write the test file; run tests and confirm they all fail.
        2. Implement `InverseBoxLaplacian.__init__`: validate args, build inverted eigenvalue table at construction time based on `null_space_handling` and `eigenvalue_tol`.
        3. Implement `_apply`: project f onto spectral basis, scale by inverted eigenvalues, sum — identical pattern to `BoxLaplacian._apply`.
        4. Run tests; confirm all pass.

2. **Phase 2: Export and reference update**
    - **Objective:** Make `InverseBoxLaplacian` part of the public API and keep the living reference accurate.
    - **Files/Functions to Modify/Create:**
        - `pygeoinf3D/pygeoinf3d/operators/__init__.py` — add `InverseBoxLaplacian`
        - `pygeoinf3D/pygeoinf3d/__init__.py` — re-export `InverseBoxLaplacian`
        - `pygeoinf3D/docs/agent-docs/references/living/pygeoinf3d-reference.md` — add entry to operators section
    - **Tests to Write:** none (export completeness checked by running the full suite)
    - **Steps:**
        1. Update both `__init__.py` files.
        2. Run the full test suite to confirm no regressions.
        3. Update the living reference.

**Open Questions (resolved)**
1. `null_space_handling` values: `'pseudoinverse'` | `'shift'` | `'strict'` — approved.
2. `eigenvalue_tol` float parameter (default `1e-10`) — approved.
3. No `alpha` scaling — omitted for consistency with `BoxLaplacian`.
