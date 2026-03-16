## Plan Complete: Bessel Sobolev Box Operator

`BesselSobolevBox` is now fully implemented, tested, documented, and archived. The operator provides shifted spectral weights $\mu_k = (\text{shift} + \lambda_k)^{s/2}$ for any real order $s$, serving as a standalone building block for future Sobolev space wrappers in pygeoinf3D. All three phases passed code review and the full test suite of 40 tests passes.

**Phases Completed:** 3 of 3
1. ✅ Phase 1: Add Core Bessel-Sobolev Box Operator
2. ✅ Phase 2: Verify Operator Action and Ladder Semantics
3. ✅ Phase 3: Document and Stabilize

**All Files Created/Modified:**
- `pygeoinf3d/operators/bessel_sobolev_box.py` — new operator implementation
- `pygeoinf3d/operators/__init__.py` — added `BesselSobolevBox` export
- `pygeoinf3d/__init__.py` — added `BesselSobolevBox` at package root
- `tests/operators/test_bessel_sobolev_box.py` — 40 tests (4 classes)
- `docs/agent-docs/references/living/pygeoinf3d-reference.md` — full operator entry, status/table updates
- `docs/agent-docs/completed-plans/bessel-sobolev-box-plan.md` — archived
- `docs/agent-docs/completed-plans/bessel-sobolev-box-phase-1-complete.md` — archived
- `docs/agent-docs/completed-plans/bessel-sobolev-box-phase-2-complete.md` — archived

**Key Functions/Classes Added:**
- `BesselSobolevBox(SpectralPowerBase3D)` — unified forward/inverse spectral operator
- `BesselSobolevBox.shift` — property returning the positive shift value
- `BesselSobolevBox.order` — property returning the real exponent
- `BesselSobolevBox._apply` — box-quadrature spectral projection apply

**Test Coverage:**
- Total tests written: 40
- All tests passing: ✅ (`40 passed in 2.22s`)
- Test classes: `TestConstructorValidation` (16), `TestEigenvalueSemantics` (10), `TestApplySemantics` (4), `TestLadderSemantics` (7), + 3 hardening regression tests

**Recommendations for Next Steps:**
- Build `Sobolev3D`: a `pygeoinf.OperatorWeightedHilbertSpace` wrapping `BesselSobolevBox` as the mass operator, providing the Bessel-Sobolev Hilbert space $H^s$ norm on a box domain.
- Extract the shared quadrature projection loop from `_apply` into `SpectralPowerBase3D` as a template method to remove the duplication between `BesselSobolevBox._apply` and `GeneralizedBoxLaplacian._apply`.
