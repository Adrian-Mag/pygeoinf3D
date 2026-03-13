## Phase 2 Complete: Gram-Aware Backend Infrastructure

Phase 2 added reusable numerical backends for Lebesgue coefficient workflows while keeping `Lebesgue3D` thin. The implementation now separates raw moments, Gram assembly, induced coefficient metrics, and coefficient solves, and exposes the distinction between the algebraic coefficient carrier and the weighted coefficient-space Hilbert structure.

**Files created/changed:**
- docs/agent-docs/references/living/pygeoinf3d-reference.md
- pygeoinf3d/__init__.py
- pygeoinf3d/representations/__init__.py
- pygeoinf3d/representations/base.py
- pygeoinf3d/representations/lebesgue.py
- pygeoinf3d/representations/backends.py
- pygeoinf3d/representations/gram.py
- pygeoinf3d/spaces/lebesgue.py
- tests/representations/test_projection_backends.py

**Functions created/changed:**
- `CoefficientRepresentation3D.coefficient_metric`
- `LebesgueCoefficientBackend.moments`
- `LebesgueCoefficientBackend.gram_matrix`
- `LebesgueCoefficientBackend.coefficient_metric`
- `LebesgueCoefficientBackend.solve_coefficients`
- `LebesgueCoefficientBackend.to_components`
- `LebesgueCoefficientBackend.from_components`
- `LebesgueRepresentation3D.coefficient_metric`
- `LebesgueRepresentation3D.gram_matrix`
- `LebesgueRepresentation3D.moments`
- `OrthonormalProjectionBackend.moments`
- `OrthonormalProjectionBackend.gram_matrix`
- `OrthonormalProjectionBackend.coefficient_metric`
- `OrthonormalProjectionBackend.solve_coefficients`
- `OrthonormalProjectionBackend.to_components`
- `OrthonormalProjectionBackend.from_components`
- `GramProjectionBackend.__init__`
- `GramProjectionBackend.moments`
- `GramProjectionBackend.gram_matrix`
- `GramProjectionBackend.coefficient_metric`
- `GramProjectionBackend.solve_coefficients`
- `GramProjectionBackend.to_components`
- `GramProjectionBackend.from_components`
- `assemble_gram`
- `gram_operators`

**Tests created/changed:**
- `TestAssembleGram`
- `TestGramOperators`
- `TestOrthonormalProjectionBackend`
- `TestGramProjectionBackend`
- `TestMomentConsistency`
- `TestLebesgueRepresentationPhase2`

**Review Status:** APPROVED with minor follow-up notes

**Git Commit Message:**
feat(lebesgue): add gram-aware coefficient backends

- add orthonormal and GramProjection backends for Lebesgue representations
- expose induced coefficient metrics separately from the algebraic carrier
- add Gram utility tests and update living reference for architecture phase 2

Plan: pygeoinf3D/docs/agent-docs/active-plans/pygeoinf3d-lebesgue-coefficient-architecture-plan.md
Phase: 2 of 5
Related: pygeoinf3D/docs/agent-docs/active-plans/pygeoinf3d-lebesgue-coefficient-architecture-plan-phase-2-complete.md