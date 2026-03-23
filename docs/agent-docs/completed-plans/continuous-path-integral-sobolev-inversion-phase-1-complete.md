## Phase 1 Complete: Add Sobolev3D Wrapper

Implemented an explicit `Sobolev3D` Hilbert space in `pygeoinf3D` as a first-class wrapper over `Lebesgue3D` plus a weight/inverse-weight operator pair. The phase now includes constructor validation, Riesz-style `to_dual` / `from_dual` behavior for primal elements, compatibility tests with `BesselSobolevBox`, and a synchronized living reference update.

**Files created/changed:**
- pygeoinf3D/pygeoinf3d/spaces/sobolev.py
- pygeoinf3D/tests/spaces/test_sobolev.py
- pygeoinf3D/pygeoinf3d/spaces/__init__.py
- pygeoinf3D/pygeoinf3d/__init__.py
- pygeoinf3D/docs/agent-docs/references/living/pygeoinf3d-reference.md

**Functions created/changed:**
- `Sobolev3D.__init__`
- `Sobolev3D.base_space`
- `Sobolev3D.function_domain`
- `Sobolev3D.weight`
- `Sobolev3D.inverse_weight`
- `Sobolev3D.dim`
- `Sobolev3D.to_dual`
- `Sobolev3D.from_dual`
- `Sobolev3D.to_components`
- `Sobolev3D.from_components`
- `Sobolev3D.inner_product`
- `Sobolev3D.norm`
- `Sobolev3D.zero`
- `Sobolev3D.add`
- `Sobolev3D.subtract`
- `Sobolev3D.multiply`
- `Sobolev3D.negative`
- `Sobolev3D.copy`
- `Sobolev3D.is_element`
- `Sobolev3D.__eq__`
- `Sobolev3D.__hash__`
- `Sobolev3D.__repr__`

**Tests created/changed:**
- `tests/spaces/test_sobolev.py::TestSobolev3DConstructor`
- `tests/spaces/test_sobolev.py::TestSobolev3DCompatibilityValidation`
- `tests/spaces/test_sobolev.py::TestSobolev3DToDual`
- `tests/spaces/test_sobolev.py::TestSobolev3DFromDual`
- `tests/spaces/test_sobolev.py::TestSobolev3DInnerProduct`
- `tests/spaces/test_sobolev.py::TestSobolev3DZeroAndArithmetic`
- `tests/spaces/test_sobolev.py::TestSobolev3DEquality`
- `tests/spaces/test_sobolev.py::TestSobolev3DComponentsNotImplemented`
- `tests/spaces/test_sobolev.py::TestSobolev3DRepr`
- `tests/spaces/test_sobolev.py::TestSobolev3DBesselSobolevBoxCompatibility`

**Review Status:** APPROVED with minor recommendations

**Git Commit Message:**
feat(spaces): add explicit Sobolev3D wrapper

- add Sobolev3D as a first-class weighted Hilbert space over Lebesgue3D
- cover Riesz round-trip, norm consistency, and BesselSobolevBox compatibility
- update public exports and living reference for the continuous Sobolev track

Plan: pygeoinf3D/docs/agent-docs/active-plans/continuous-path-integral-sobolev-inversion-plan.md
Phase: 1 of 5
Related: pygeoinf3D/docs/agent-docs/active-plans/continuous-path-integral-sobolev-inversion-phase-1-complete.md
