## Phase 2 Complete: Implement Dual And Adjoint Correctly

Added explicit dual and adjoint mappings to `SOLAOperator3D` so the operator now respects arbitrary finite-dimensional Hilbert codomain inner products while staying basis-free on the domain side. The implementation uses codomain-dual coordinates for the dual map, uses the codomain Riesz map for the adjoint, and adds coverage for Euclidean and mass-weighted codomains together with a defensive zero-measurement guard.

**Files created/changed:**
- pygeoinf3d/operators/sola.py
- tests/operators/test_sola.py
- docs/agent-docs/references/living/pygeoinf3d-reference.md

**Functions created/changed:**
- `SOLAOperator3D.__init__`
- `SOLAOperator3D._dual_mapping`
- `SOLAOperator3D._adjoint_mapping`
- `SOLAOperator3D._forward`
- `_make_mass_weighted_codomain`

**Tests created/changed:**
- `test_sola_dual_uses_codomain_dual_coordinates`
- `test_sola_dual_single_measurement`
- `test_sola_adjoint_matches_lebesgue_pairing`
- `test_sola_mass_weighted_codomain_adjoint_identity`
- `test_sola_rejects_zero_measurements`

**Review Status:** APPROVED

**Git Commit Message:**
feat(operators): implement SOLAOperator3D dual and adjoint mappings

- Add explicit codomain-aware dual mapping using codomain.dual.to_components
- Add explicit adjoint mapping via codomain.to_dual and Lebesgue3D.from_dual
- Extend SOLA tests for Euclidean and mass-weighted codomain adjoint identities

Plan: pygeoinf3D/docs/agent-docs/active-plans/pygeoinf3d-sola-operator-plan.md
Phase: 2 of 4
Related: pygeoinf3D/docs/agent-docs/active-plans/pygeoinf3d-sola-operator-plan-phase-2-complete.md