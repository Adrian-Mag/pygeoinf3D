# Plan: Scalable SOLA via Push-Through + Streaming K

Reformulate the SOLA solver to work in the fixed-size model/quadrature space
($N_\text{quad} \times N_\text{quad}$) instead of the growing data space
($N_d \times N_d$), add streaming kernel evaluation so K is never fully
materialized in RAM for large $N_d$, and expose clean methods for Function3D
reconstruction. Target: $N_d = 10{,}000+$ on a laptop in under 10 minutes.

## Mathematics

Define $\tilde{K} = K \operatorname{diag}(\sqrt{w})$ (shape $N_d \times N_q$, subscripts G or T for forward/target).
The model-space normal matrix is $M = \tilde{K}_G^T \tilde{K}_G + \sigma^2 I$ (shape $N_q \times N_q$, always small).
By the push-through identity:

- **Solve**: $(GG^* + \sigma^2 I)^{-1} d = \frac{1}{\sigma^2}(d - \tilde{K}_G M^{-1} \tilde{K}_G^T d)$
- **SOLA matrix**: $X = \tilde{K}_T M^{-1} \tilde{K}_G^T$ (shape $N_p \times N_d$)
- **Reconstruction**: $\hat{m}(x) = \sum_i q_i k_i(x)$ where $q$ comes from Solve

Both $M$ and $X$ can be assembled without materializing the full $K_G$ matrix,
by streaming kernel rows in chunks.

## Phases (5 phases)

### Phase 1: `evaluate_batch` on providers + chunked model-space Gram in `VolumeInteractionBackend`

- **Objective:** Add `evaluate_batch(i_start, i_end, pts)` to `FunctionProvider3D`
  (default: loop over `get_function`; `GaussianProvider3D` overrides with vectorized slice).
  Add `build_model_gram(noise_std, chunk_size=None)` and `normal_vector(d, chunk_size=None)`
  and `kernel_matvec(v, chunk_size=None)` to `VolumeInteractionBackend`.

- **Files/Functions to Modify/Create:**
  - `pygeoinf3D/pygeoinf3d/providers/base.py` — add `evaluate_batch` to `FunctionProvider3D`
  - `pygeoinf3D/pygeoinf3d/providers/function_providers/gaussian.py` — override `evaluate_batch` in `GaussianProvider3D`
  - `pygeoinf3D/pygeoinf3d/providers/function_providers/boxcar.py` — add `evaluate_batch` override
  - `pygeoinf3D/pygeoinf3d/providers/function_providers/bump.py` — add `evaluate_batch` override
  - `pygeoinf3D/pygeoinf3d/solvers/volume_interactions.py` — add `build_model_gram`, `normal_vector`, `kernel_matvec`, `_default_chunk_size`

- **Tests to Write:**
  - `test_evaluate_batch_matches_evaluate_all_gaussian` — `GaussianProvider3D.evaluate_batch(0, N, pts)` matches `evaluate_all`
  - `test_evaluate_batch_slice_gaussian` — `evaluate_batch(1, 2, pts)` matches row 1 of `evaluate_all`
  - `test_evaluate_batch_default_fallback` — using the loop-based `_LinearMockProvider`, verify batch works
  - `test_build_model_gram_matches_explicit` — for small N_d, $M = K_G^T W K_G + \sigma^2 I$ matches explicitly built K
  - `test_build_model_gram_chunked_equals_full` — chunk_size=1, chunk_size=3, chunk_size=N_d all agree
  - `test_normal_vector_matches_explicit` — $\tilde{K}_G^T d$ chunked matches `sqrt_w * K.T @ d`
  - `test_kernel_matvec_matches_explicit` — $K_G v$ chunked matches `K @ v`
  - All in `pygeoinf3D/tests/solvers/test_volume_interaction_backend.py`

- **Steps:**
  1. Write all failing tests listed above
  2. Add `evaluate_batch` default to `FunctionProvider3D` (calls `evaluate_all` and slices rows)
  3. Override `evaluate_batch` in `GaussianProvider3D` with vectorized slice on `_centers[i:j]` and `_sigmas[i:j]`
  4. Override `evaluate_batch` in `BoxcarProvider3D` and `BumpProvider3D` similarly
  5. Add `_default_chunk_size(N_d, memory_budget_bytes=256*1024**2)` to `VolumeInteractionBackend` — returns `max(1, budget // (N_q * 8))` where `N_q = len(self._weights)`
  6. Add `build_model_gram(noise_std, chunk_size=None)` to `VolumeInteractionBackend`:
     ```
     sqrt_w = sqrt(weights); M = sigma^2 * I_Nq
     for each chunk of rows i:j from provider:
         Kw_chunk = provider.evaluate_batch(i, j, pts) * sqrt_w  # (B, N_q)
         M += Kw_chunk.T @ Kw_chunk  # symmetric rank-B update
     ```
     Returns `(M, cho_factor(M), sqrt_w)`.
  7. Add `normal_vector(d, chunk_size=None)` — computes $\sqrt{w} \odot (K_G^T d)$ streaming (or from cache if available)
  8. Add `kernel_matvec(v, chunk_size=None)` — computes $K_G v$ (result shape $N_d$) streaming
  9. Run tests; verify all pass

---

### Phase 2: Push-through solve methods on `DenseVolumeSolaSolver`

- **Objective:** Add `solve(d, noise_std)`, `reconstruct(d, noise_std)`, and
  `compute_sola_matrix(target_solver, noise_std)` to `DenseVolumeSolaSolver`,
  all using the model-space push-through with lazy-cached $M$ Cholesky.

- **Files/Functions to Modify/Create:**
  - `pygeoinf3D/pygeoinf3d/solvers/dense_volume_sola.py` — add three methods + `_ensure_model_cho` cache

- **Tests to Write:**
  - `test_solve_matches_data_space_cholesky` — `solver.solve(d, sigma)` vs `cho_solve(GG*+sigma^2 I, d)` at small N_d, rtol=1e-10
  - `test_solve_is_chainable` — call solve twice, cache is reused (same `_model_M_cho` object)
  - `test_reconstruct_returns_function3d` — returned object is callable, evaluates at arbitrary points
  - `test_reconstruct_matches_sola_estimate` — `solver.reconstruct(G(m_true), sigma)(pts)` is approximately `m_true(pts)`
  - `test_compute_sola_matrix_shape` — shape is `(N_p, N_d)`
  - `test_compute_sola_matrix_matches_cholesky_pipeline` — at small N_d, `X_mat @ d ≈ T_op @ G_inv(d)` with rtol=1e-10
  - `test_compute_sola_matrix_chunked_equals_full` — same result with chunk_size=1 and chunk_size=N_d
  - All in `pygeoinf3D/tests/solvers/test_dense_volume_sola.py`

- **Steps:**
  1. Write all failing tests
  2. Add `_model_M_cho = None`, `_model_sqrt_w = None`, `_model_noise_std = None` to `__init__` init
  3. Add `_ensure_model_cho(noise_std, chunk_size=None)`:
     - If not cached or noise_std changed, call `self._backend.build_model_gram(noise_std, chunk_size)`
     - Cache result in instance variables
  4. Add `solve(d, noise_std, chunk_size=None) -> np.ndarray`:
     - Calls `_ensure_model_cho`
     - `v = backend.normal_vector(d, chunk_size)` — $\tilde{K}_G^T d$
     - `z = cho_solve(M_cho, v)` — $M^{-1} \tilde{K}_G^T d$
     - `Kz = backend.kernel_matvec(sqrt_w * z, chunk_size)` — $K_G (\sqrt{w} \odot z)$
     - Returns `(d - Kz) / noise_std**2`
  5. Add `reconstruct(d, noise_std, chunk_size=None) -> Function3D`:
     - Calls `solve(d, noise_std)` → `q`
     - Returns `Function3D` whose callable is `lambda x: backend._provider.evaluate_all(x).T @ q`
     - Note: for large $N_d$, warn once that evaluation cost is $O(N_d \times n_x)$
  6. Add `compute_sola_matrix(target_solver, noise_std, chunk_size=None) -> np.ndarray`:
     - Calls `_ensure_model_cho`
     - Gets $\tilde{K}_T$ from `target_solver.backend`
     - Solves $M Z_T = \tilde{K}_T^T$ → $Z_T$ shape $(N_q, N_p)$
     - Computes $H = Z_T^T * \sqrt{w}[\text{None},:]$ shape $(N_p, N_q)$
     - Assembles $X[:, i:j] = H @ K_{G,i:j}^T$ in chunks → shape $(N_p, N_d)$
     - Returns $X$
  7. Run tests; verify all pass

---

### Phase 3: Update `resolve()` + lazy/streaming init

- **Objective:**
  (a) Update `resolve(target_K_row, noise_std)` to use push-through (requires `noise_std` parameter; add deprecation path for old signature).
  (b) Add `memory_budget_bytes` constructor parameter; when $N_d \times N_q \times 8 >$ budget, skip K caching and all push-through paths use chunked evaluation.
  (c) `forward_operator()` warns when K is not cached.

- **Files/Functions to Modify/Create:**
  - `pygeoinf3D/pygeoinf3d/solvers/dense_volume_sola.py` — modify `__init__`, `resolve`
  - `pygeoinf3D/pygeoinf3d/solvers/volume_interactions.py` — add `cache_k` parameter

- **Tests to Write:**
  - `test_resolve_push_through_matches_old` — with noise_std>0, new `resolve(row, noise_std)` matches old at small N_d
  - `test_large_nd_no_k_cache` — construct solver with memory_budget=1 (forces no cache); verify `_backend._K_cache is None`; verify `solve` still works correctly
  - `test_forward_operator_warns_when_no_cache` — `solver.forward_operator(space)` emits UserWarning when K not cached
  - All in `pygeoinf3D/tests/solvers/test_dense_volume_sola.py`

- **Steps:**
  1. Write failing tests
  2. Add `cache_k: bool` field to `VolumeInteractionBackend.__init__` (default True); when False, `interaction_matrix()` recomputes each time (no caching) and warns on first call
  3. Add `memory_budget_bytes: int = 512 * 1024**2` parameter to `DenseVolumeSolaSolver.__init__`; compute `k_bytes = N_d * N_q * 8`; if `k_bytes > memory_budget_bytes`, set `cache_k=False` on backend
  4. Move Cholesky from `__init__` into a lazy property `_data_cho` (only computed on first `resolve` call when old signature used, or when K is cached)
  5. Update `resolve(target_K_row, noise_std=None)`:
     - If `noise_std` is provided: use push-through via `solve(cross_gram_vector, noise_std)` ... actually note that `resolve` currently takes a target_K_row (quadrature-space vector) and computes cross-gram internally. Keep this: compute cross = `K @ (w * target)` streaming, then use Woodbury.
     - If `noise_std` is None (legacy): use old data-space Cholesky path (with deprecation warning)
  6. Add UserWarning in `forward_operator()` when `_backend._K_cache is None`
  7. Run tests; verify all pass

---

### Phase 4: Update `sola_noise.py` to use new API

- **Objective:** Replace the `CholeskySolver` + `Lambda = G @ G.adjoint` block with
  `solver_G.compute_sola_matrix(solver_T, noise_std)`. Update reconstruction to use
  `solver_G.reconstruct(d, noise_std)`. Update unimodularity section. Add `--n-d` CLI flag.

- **Files/Functions to Modify/Create:**
  - `pygeoinf3D/demos/box/sola/sola_noise.py`

- **Tests to Write:**
  - Manual: run `python sola_noise.py --no-viz` with N_d=100, verify property means/stds match old output to 4 sig figs.

- **Steps:**
  1. Update imports: remove `CholeskySolver`, add nothing (new methods on solver)
  2. Add `--n-d INT` CLI argument (default 100)
  3. Replace SOLA construction section:
     ```python
     # OLD:
     cholesky_solver = CholeskySolver(...)
     Lambda  = G @ G.adjoint
     W_inv   = cholesky_solver(Lambda + gaussian_D.covariance)
     G_inv   = G.adjoint @ W_inv
     X       = T_op @ G_inv

     # NEW:
     X_mat = solver_G.compute_sola_matrix(solver_T, noise_std)
     ```
  4. Update reconstruction:
     ```python
     # OLD: m_hat = G_inv(d_tilde)
     # NEW:
     m_hat = solver_G.reconstruct(d_tilde, noise_std)
     ```
  5. Update property inference: `X` is now a numpy matrix $(N_p \times N_d)$.
     Replace `property_measure = gaussian_D.affine_mapping(operator=X)` with direct computation:
     ```python
     p_mean    = X_mat @ d_tilde
     P_cov_mat = noise_std**2 * X_mat @ X_mat.T
     p_std     = np.sqrt(np.diag(P_cov_mat))
     ```
  6. Update unimodularity section: `u = solver_G.solve(v, noise_std)`;
     `X(v)` becomes `X_mat @ v`; `np.dot(v, u)` stays as-is.
     Correction matrix is still a rank-1 update to `X_mat`.
     `X_unimodular` becomes a numpy array.
     Update property inference for unimodular case similarly.
  7. Run `python sola_noise.py --no-viz` and verify output matches previous to 4 sig figs.

---

### Phase 5: Benchmark script

- **Objective:** Verify correctness and timing at $N_d \in \{100, 500, 1000, 5000, 10000\}$.

- **Files/Functions to Modify/Create:**
  - `pygeoinf3D/rough_work/sola_scale_benchmark.py` — new script

- **Tests to Write:** (benchmark, not unit tests)

- **Steps:**
  1. Write benchmark script:
     - For each N_d in [100, 500, 1000, 5000, 10000]:
       - Build solver (record init time)
       - Run `compute_sola_matrix` (record time)
       - Compute p_mean and p_cov_mat (record time)
       - Report: total time, peak memory (via tracemalloc)
       - Report relative residual and property error vs true model
  2. Run locally with `inferences3` env
  3. Print summary table and verify quality doesn't degrade with N_d

---

## Open Questions (resolved before implementation)

1. `evaluate_batch` for all providers: add to `GaussianProvider3D`, `BoxcarProvider3D`, `BumpProvider3D`; default in `FunctionProvider3D` falls back to `evaluate_all()[i:j]`.
2. `forward_operator()` warns (once, via `warnings.warn`) when K is not cached.
3. `resolve(target_K_row, noise_std=None)` — old signature (noise_std=None) uses legacy Cholesky with deprecation warning; new signature (noise_std provided) uses push-through.
