## Phase 1 Complete: Representation Boundary and API Contracts

Phase 1 introduced the representation boundary for `Lebesgue3D` without changing its pairing-first semantics. `Lebesgue3D` is still basis-free by default, but it can now delegate coefficient operations to an attached finite-dimensional representation object, and the new representation package establishes the abstraction boundary for later Gram-aware backends.

**Files created/changed:**
- docs/agent-docs/active-plans/pygeoinf3d-lebesgue-coefficient-architecture-plan.md
- docs/agent-docs/references/living/pygeoinf3d-reference.md
- pygeoinf3d/__init__.py
- pygeoinf3d/spaces/lebesgue.py
- pygeoinf3d/representations/__init__.py
- pygeoinf3d/representations/base.py
- pygeoinf3d/representations/lebesgue.py
- tests/spaces/test_lebesgue.py
- tests/representations/__init__.py
- tests/representations/test_lebesgue_representation.py

**Functions created/changed:**
- `Lebesgue3D.__init__`
- `Lebesgue3D.dim`
- `Lebesgue3D.to_components`
- `Lebesgue3D.from_components`
- `Lebesgue3D.__repr__`
- `CoefficientRepresentation3D.coefficient_space`
- `LebesgueRepresentation3D.__init__`
- `LebesgueRepresentation3D.space`
- `LebesgueRepresentation3D.dim`
- `LebesgueRepresentation3D.to_components`
- `LebesgueRepresentation3D.from_components`
- `LebesgueRepresentation3D.provider`
- `LebesgueRepresentation3D.backend`
- `LebesgueRepresentation3D.__repr__`

**Tests created/changed:**
- `TestLebesgue3DWithRepresentation`
- `TestCoefficientRepresentation3DABC`
- `TestLebesgueRepresentation3DConstruction`
- `TestLebesgueRepresentation3DDim`
- `TestLebesgueRepresentation3DDelegation`
- `TestCoefficientRepresentation3DCoeffSpace`

**Review Status:** APPROVED

**Git Commit Message:**
feat(lebesgue): add representation boundary for coefficient workflows

- add CoefficientRepresentation3D and LebesgueRepresentation3D abstractions
- delegate Lebesgue3D component maps to optional attached representation
- add phase 1 tests and update living reference for representation package

Plan: pygeoinf3D/docs/agent-docs/active-plans/pygeoinf3d-lebesgue-coefficient-architecture-plan.md
Phase: 1 of 5
Related: pygeoinf3D/docs/agent-docs/completed-plans/pygeoinf3d-lebesgue-coefficient-architecture-plan-phase-1-complete.md