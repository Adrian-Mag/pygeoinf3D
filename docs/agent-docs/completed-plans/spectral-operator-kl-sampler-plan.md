## Plan: SpectralOperator3D ABC + KLSampler3D

Introduce a formal `SpectralOperator3D` ABC that surfaces `get_eigenvalue`, `get_eigenfunction`, and `n_modes` on any spectrally-implemented operator, retrofit `BoxLaplacian` and `InverseBoxLaplacian` to satisfy it, then implement a generic `KLSampler3D` in a new `sampling/` sub-package. Together these are the missing pieces needed to construct Gaussian prior measures for a 3D PLI notebook.

**Phases (3)**

1. **Phase 1: `SpectralOperator3D` ABC + operator retrofits**
    - **Objective:** Formalise the spectral protocol and make both existing operators satisfy it
    - **Files/Functions to Modify/Create:**
      - Create `pygeoinf3d/operators/base.py` — `SpectralOperator3D(LinearOperator, ABC)` requiring `get_eigenvalue(i)`, `get_eigenfunction(i)`, `n_modes` property
      - Modify `laplacian_box.py` — inherit `SpectralOperator3D`; add forwarding methods/property to `_provider`
      - Modify `inverse_laplacian_box.py` — inherit `SpectralOperator3D`; `get_eigenvalue(i)` returns `_modes[i][0]`, `get_eigenfunction(i)` returns `_modes[i][1]`
      - Modify `operators/__init__.py` — export `SpectralOperator3D`
    - **Tests to Write:**
      - In `test_box_laplacian.py`: `test_is_spectral_operator`, `test_get_eigenvalue_matches_provider`, `test_get_eigenfunction_matches_provider`, `test_n_modes`
      - In `test_inverse_box_laplacian.py`: `test_is_spectral_operator`, `test_get_eigenvalue_returns_inverted`, `test_get_eigenfunction_same_as_provider`, `test_n_modes_matches_provider`
    - **Steps:**
        1. Write tests (fail — `SpectralOperator3D` does not exist yet)
        2. Create `operators/base.py`
        3. Retrofit `BoxLaplacian` and `InverseBoxLaplacian`
        4. Run tests — all pass

2. **Phase 2: `KLSampler3D`**
    - **Objective:** Implement the KL-expansion sampler generically over any `SpectralOperator3D`
    - **Files/Functions to Modify/Create:**
      - Create `pygeoinf3d/sampling/__init__.py`
      - Create `pygeoinf3d/sampling/kl_sampler.py` — `KLSampler3D`
      - Create `tests/sampling/__init__.py`
      - Create `tests/sampling/test_kl_sampler.py`
    - **API:**
      - `KLSampler3D(operator: SpectralOperator3D, *, n_modes: int, mean: Function3D | None, rng: np.random.Generator | None)`
      - `sample() → Function3D` — $u = m + \sum_{k<K} \sqrt{\lambda_k}\,\xi_k\,\Phi_k$
      - `samples(n: int) → list[Function3D]`
      - `covariance_factor` (property) — `LinearOperator: EuclideanSpace(K) → Lebesgue3D`
    - **Tests to Write:**
      - `test_construct_from_inverse_laplacian_dirichlet`
      - `test_n_modes_bad_value_raises`
      - `test_n_modes_exceeds_operator_raises`
      - `test_sample_returns_function3d`
      - `test_sample_batch_shape`
      - `test_samples_are_distinct`
      - `test_covariance_factor_is_linear_operator`
      - `test_covariance_factor_domain_euclidean`
      - `test_covariance_factor_codomain_is_space`
      - `test_covariance_factor_eigenfunction_reconstruction_mode0`
      - `test_covariance_factor_eigenfunction_reconstruction_mode1`
      - `test_zero_mode_neumann_pseudoinverse_vanishes_in_covariance_factor`
      - `test_sample_mean_is_added` (statistical, fixed seed)
    - **Steps:**
        1. Write tests (fail)
        2. Implement `KLSampler3D`
        3. Run tests — all pass

3. **Phase 3: Top-level exports + full suite + reference**
    - **Objective:** Make `KLSampler3D` and `SpectralOperator3D` part of the public API; verify no regressions; update reference
    - **Files/Functions to Modify/Create:**
      - `pygeoinf3d/__init__.py` — export `KLSampler3D`, `SpectralOperator3D`
      - `docs/agent-docs/references/living/pygeoinf3d-reference.md` — add sampling section, update operators section
    - **Tests to Write:** none (imports verified by full test suite)
    - **Steps:**
        1. Update `__init__.py`
        2. Run full test suite — no regressions
        3. Update living reference

**Open Questions** (resolved before implementation)
1. `n_modes` — required explicit integer (no energy_tol for now)
2. `covariance_factor` — property (not a method)
3. No `alpha` scaling on the sampler (leave to user / operator construction)
