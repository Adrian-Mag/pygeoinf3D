## Plan: pygeoinf3D Generalized Box Laplacian

Add a new `GeneralizedBoxLaplacian` spectral operator family that represents integer powers of the positive box operator $(-\alpha \Delta)^s$ on `Lebesgue3D`, with support for both positive and negative integer powers including the identity case `power = 0`. The design also introduces a broader non-box-specific spectral-power base so later operators can reuse the same spectral functional-calculus machinery, keeps `BoxLaplacian` and `InverseBoxLaplacian` as stable public entry points, makes mean-zero projection the default way to handle Neumann zero modes for negative powers, and leaves shifted/Bessel-style variants out of scope for this first pass.

**Phases 5**
1. **Phase 1: Operator Semantics and Compatibility Tests**
    - **Objective:** Lock down mathematically consistent integer-power semantics and write regression tests that preserve the current Laplacian and inverse Laplacian behavior while explicitly supporting the identity case.
    - **Files/Functions to Modify/Create:** `pygeoinf3D/tests/operators/test_box_laplacian.py`, `pygeoinf3D/tests/operators/test_inverse_box_laplacian.py`, new generalized-operator test file under `pygeoinf3D/tests/operators/`
    - **Tests to Write:** positive-power eigenmode tests, negative-power pseudoinverse tests for Neumann zero mode, identity semantics for `power = 0`, equivalence tests showing `power=1` matches `BoxLaplacian` and `power=-1` matches `InverseBoxLaplacian`, invalid-argument tests for non-integer powers and non-positive `alpha`
    - **Steps:**
        1. Write failing tests for integer spectral-power behavior using the existing tensor-product eigenbasis.
        2. Write regression tests that capture the current `BoxLaplacian` and `InverseBoxLaplacian` semantics exactly.
        3. Encode the chosen convention that the generalized family acts on the positive operator $(-\alpha \Delta)$ rather than raw $(\alpha \Delta)$.
        4. Make mean-zero projection the documented and tested default for near-zero modes when `power < 0`.

2. **Phase 2: Implement Spectral-Power Base and `GeneralizedBoxLaplacian` Core**
    - **Objective:** Introduce a reusable non-box-specific spectral-power base and implement the generalized box operator for integer powers with explicit zero-mode handling and spectral-interface compatibility.
    - **Files/Functions to Modify/Create:** new base/operator modules in `pygeoinf3D/pygeoinf3d/operators/`, `pygeoinf3D/pygeoinf3d/operators/__init__.py`, `pygeoinf3D/pygeoinf3d/__init__.py`
    - **Tests to Write:** constructor validation tests, `get_eigenvalue` transformation tests, `get_eigenfunction` forwarding tests, self-adjointness smoke tests, apply-on-eigenfunction tests across positive, zero, and negative powers, base-class delegation tests where appropriate
    - **Steps:**
        1. Introduce a reusable spectral-power base class that transforms the spectrum of an underlying `SpectralOperator3D` using integer powers of a positive eigenvalue sequence.
        2. Implement `GeneralizedBoxLaplacian(space, provider, *, power, alpha=1.0, eigenvalue_tol=...)` on top of that base.
        3. Restrict `power` to integers only and fail fast for non-integer inputs.
        4. Compute operator eigenvalues as $(\alpha \lambda_k)^s` for integer `s`, with pseudoinverse-style suppression of modes satisfying `|\lambda_k| < eigenvalue_tol` when `s < 0`.
        5. Support `power = 0` explicitly as the identity on `Lebesgue3D`.
        6. Preserve the tensor-product provider attribute so `KLSampler3D` and current fast-path detection continue to work.

3. **Phase 3: Refactor Existing Wrappers Onto the Generalized Engine**
    - **Objective:** Keep the current public APIs intact while removing duplicated spectral-power logic.
    - **Files/Functions to Modify/Create:** `pygeoinf3D/pygeoinf3d/operators/laplacian_box.py`, `pygeoinf3D/pygeoinf3d/operators/inverse_laplacian_box.py`, corresponding test files
    - **Tests to Write:** wrapper-equivalence tests, repr/default-behavior regression tests, backward-compatible nullspace tests for `InverseBoxLaplacian`
    - **Steps:**
        1. Rework `BoxLaplacian` into a thin semantic wrapper equivalent to `GeneralizedBoxLaplacian(..., power=1, alpha=1.0)`.
        2. Rework `InverseBoxLaplacian` into a thin semantic wrapper equivalent to `GeneralizedBoxLaplacian(..., power=-1, alpha=1.0)` with mean-zero pseudoinverse handling.
        3. Preserve constructor signatures and documented user-facing behavior unless a change is strictly required.
        4. Verify that existing notebooks and operator consumers keep working unchanged.

4. **Phase 4: KL and Sampling Integration Checks**
    - **Objective:** Ensure the new generalized operator behaves correctly when used as a covariance-like spectral operator and does not break current KL workflows.
    - **Files/Functions to Modify/Create:** `pygeoinf3D/tests/sampling/test_kl_sampler.py`, optionally focused tests in `pygeoinf3D/tests/test_plot.py` if fast-path assumptions need coverage
    - **Tests to Write:** KL compatibility tests for negative powers with suppressed zero modes, tests ensuring tensor-product fast-path detection still works, tests that covariance-like operators expose nonnegative eigenvalues when required for sampling
    - **Steps:**
        1. Verify that `GeneralizedBoxLaplacian(..., power=-1)` is accepted by `KLSampler3D` in the same way as the current inverse operator.
        2. Add explicit tests for Neumann zero-mode suppression under KL sampling.
        3. Confirm that positive-power operators are not accidentally treated as valid covariance operators in sampling contexts without clear eigenvalue semantics.
        4. Preserve the current private fast-path bridge through `_provider`.

5. **Phase 5: Documentation and Architecture Alignment**
    - **Objective:** Document the new operator family clearly and position both the reusable spectral-power base and the box-specific specialization relative to the still-active representation/backend architecture work.
    - **Files/Functions to Modify/Create:** `pygeoinf3D/README.md`, `pygeoinf3D/docs/agent-docs/references/living/pygeoinf3d-reference.md`, operator module docstrings, optional demo notes
    - **Tests to Write:** none beyond doc example validation if examples are added
    - **Steps:**
        1. Update the living reference with the new spectral-power base, the generalized box operator, wrapper relationships, and integer-power limitation.
        2. State explicitly that the public convention is powers of the positive operator $(-\alpha \Delta)$.
        3. Document that `power = 0` is supported as the identity within the same operator family.
        4. Document that bounded-domain fractional/integral-style semantics are out of scope for this phase because only integer powers are supported.
        5. Note that future operator refactors should eventually align with the active Lebesgue coefficient architecture plan rather than duplicating projection internals indefinitely.

    **Resolved Decisions 2**
    1. `power = 0` is supported explicitly as the identity operator on `Lebesgue3D`.
    2. The first implementation introduces both `GeneralizedBoxLaplacian` and a broader non-box-specific spectral-power base class for later reuse.