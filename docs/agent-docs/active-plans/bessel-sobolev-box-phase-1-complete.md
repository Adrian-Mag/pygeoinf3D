## Phase 1 Complete: Add Core Bessel-Sobolev Box Operator

Implemented `BesselSobolevBox` as a standalone box-domain spectral operator in pygeoinf3D, with unified eigenvalue semantics `mu_k = (shift + lambda_k) ** (order / 2)` for real-valued `order`. The class is exported through the operators package and package root, focused tests are in place, and the living reference has been minimally synced to reflect the new public operator surface.

**Files created/changed:**
- `pygeoinf3D/pygeoinf3d/operators/bessel_sobolev_box.py`
- `pygeoinf3D/pygeoinf3d/operators/__init__.py`
- `pygeoinf3D/pygeoinf3d/__init__.py`
- `pygeoinf3D/tests/operators/test_bessel_sobolev_box.py`
- `pygeoinf3D/docs/agent-docs/references/living/pygeoinf3d-reference.md`

**Functions created/changed:**
- `BesselSobolevBox.__init__`
- `BesselSobolevBox.shift`
- `BesselSobolevBox.order`
- `BesselSobolevBox._apply`
- `BesselSobolevBox.__repr__`

**Tests created/changed:**
- `TestConstructorValidation`
- `TestEigenvalueSemantics`
- `TestApplySemantics`
- Added finite-shift regression tests for `np.nan` and `np.inf`

**Review Status:** APPROVED

**Git Commit Message:**
```text
feat(operators): add BesselSobolevBox spectral operator

- Implement shifted box spectral operator with real-valued order/2 weights
- Add focused tests for validation, eigenvalues, and apply semantics
- Export operator and sync living reference with new public surface

Plan: pygeoinf3D/docs/agent-docs/active-plans/bessel-sobolev-box-plan.md
Phase: 1 of 3
Related: pygeoinf3D/docs/agent-docs/active-plans/bessel-sobolev-box-phase-1-complete.md
```
