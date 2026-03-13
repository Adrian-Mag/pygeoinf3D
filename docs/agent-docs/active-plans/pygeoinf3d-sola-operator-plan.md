## Plan: pygeoinf3D SOLA Operator

Implement a representation-independent SOLA-style measurement operator for pygeoinf3D as a true subclass of pygeoinf.LinearOperator. The operator must remain continuous-first on the domain side, work with arbitrary finite-dimensional Hilbert codomains through explicit forward, dual, and adjoint mappings, and begin with volume-kernel measurements whose adjoints are Riesz-representable in Lebesgue3D.

**Phases 4**
1. **Phase 1: Add Volume SOLA Core**
    - **Objective:** Introduce a LinearOperator subclass for stacked volume-kernel measurements without using Lebesgue3D component machinery.
    - **Files/Functions to Modify/Create:** `pygeoinf3d/operators/sola.py`, `pygeoinf3d/operators/__init__.py`, `pygeoinf3d/__init__.py`, `tests/operators/test_sola.py`
    - **Tests to Write:** `test_sola_forward_volume_kernels_matches_analytic_integrals`, `test_sola_forward_respects_custom_codomain_realization`, `test_sola_rejects_mismatched_kernel_counts`
    - **Steps:**
        1. Write failing tests for forward mapping from Lebesgue3D into finite-dimensional codomains using VolumeKernel3D measurements.
        2. Implement `SOLAOperator3D` as a `LinearOperator` subclass with explicit constructor validation and forward mapping only through form evaluations plus codomain realization.
        3. Run operator tests and adjust exports until the new public API passes.

2. **Phase 2: Implement Dual And Adjoint Correctly**
    - **Objective:** Define dual and adjoint mappings that respect arbitrary codomain inner products while remaining valid for volume-kernel measurements.
    - **Files/Functions to Modify/Create:** `pygeoinf3d/operators/sola.py`, `tests/operators/test_sola.py`
    - **Tests to Write:** `test_sola_dual_uses_codomain_dual_coordinates`, `test_sola_adjoint_matches_lebesgue_pairing`, `test_sola_mass_weighted_codomain_adjoint_identity`
    - **Steps:**
        1. Write failing tests for dual and adjoint behavior with plain Euclidean and mass-weighted codomains.
        2. Implement explicit `dual_mapping` from codomain dual coordinates and explicit `adjoint_mapping` via codomain Riesz map followed by domain `from_dual`.
        3. Run the focused operator tests to confirm the adjoint identity holds in both codomain geometries.

3. **Phase 3: Add Measurement Providers And Lazy Access**
    - **Objective:** Support provider-backed measurement families on the dual side without introducing basis providers or representation coupling.
    - **Files/Functions to Modify/Create:** `pygeoinf3d/operators/sola.py`, `pygeoinf3d/providers/__init__.py`, optionally `pygeoinf3d/providers/base.py`, `tests/operators/test_sola.py`
    - **Tests to Write:** `test_sola_accepts_volume_kernel_provider`, `test_sola_lazy_provider_access_matches_list_kernels`, `test_sola_provider_cache_returns_stable_kernels`
    - **Steps:**
        1. Write failing tests for indexed measurement access through provider-backed volume kernels.
        2. Introduce a minimal provider abstraction or provider protocol for indexed kernels and implement lazy retrieval in the operator.
        3. Run operator tests to verify list-backed and provider-backed kernels agree exactly.

4. **Phase 4: Document Extension Boundaries**
    - **Objective:** Record the operator API, representation-independence guarantees, and the future boundary between generic dual measurements and Riesz-representable adjoints.
    - **Files/Functions to Modify/Create:** `pygeoinf3d/__init__.py`, `pygeoinf3d/operators/__init__.py`, `docs/agent-docs/references/living/pygeoinf3d-reference.md`, optionally `README.md`
    - **Tests to Write:** `test_sola_public_reexports_available`
    - **Steps:**
        1. Write or extend a failing export test that locks the public API for the new operator.
        2. Update the living reference and public exports to describe the operator, its codomain-general dual/adjoint semantics, and the current restriction to volume-kernel adjoints.
        3. Run the export and operator tests to confirm documentation and public API remain synchronized.

**Open Questions 1**
1. Future path and surface measurements should keep the same forward and dual API, but adjoint support must remain explicitly limited to spaces whose `from_dual` can represent those measurements.