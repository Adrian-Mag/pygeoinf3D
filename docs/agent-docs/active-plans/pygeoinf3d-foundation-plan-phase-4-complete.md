## Phase 4 Complete: Sobolev and Bessel-Type Operators on Box Domains

Phase 4 was refactored to match the continuous-first design requirement rather than the earlier coefficient-space shortcut. The public `Sobolev3D` and `BesselOperator3D` APIs now operate on `Function3D` objects, while the old finite coefficient-space machinery survives only as explicit secondary APIs (`SpectralSobolev3D` and `SpectralBesselOperator3D`).

**Files created/changed:**
- pygeoinf3D/pygeoinf3d/__init__.py
- pygeoinf3D/pygeoinf3d/operators/__init__.py
- pygeoinf3D/pygeoinf3d/operators/bessel.py
- pygeoinf3D/pygeoinf3d/providers/tensor_product.py
- pygeoinf3D/pygeoinf3d/spaces/__init__.py
- pygeoinf3D/pygeoinf3d/spaces/forms.py
- pygeoinf3D/pygeoinf3d/spaces/sobolev.py
- pygeoinf3D/pygeoinf3d/geometries/box/domain.py
- pygeoinf3D/tests/operators/test_bessel.py
- pygeoinf3D/tests/spaces/test_sobolev.py
- pygeoinf3D/docs/agent-docs/references/living/pygeoinf3d-reference.md

**Functions created/changed:**
- `Sobolev3D.__init__`
- `Sobolev3D.to_dual`
- `Sobolev3D.from_dual`
- `Sobolev3D.to_components`
- `Sobolev3D.from_components`
- `Sobolev3D.project`
- `Sobolev3D.inner_product`
- `Sobolev3D._project_to_coeffs`
- `Sobolev3D._reconstruct`
- `SpectralSobolev3D.from_dual`
- `SobolevKernel3D.__init__`
- `SobolevKernel3D._mapping_impl`
- `BesselOperator3D.__init__`
- `BesselOperator3D._apply`
- `SpectralBesselOperator3D.__init__`
- `SpectralBesselOperator3D._apply`
- `bessel_direct`
- `bessel_inverse`
- `BoxDomain.sample_boundary`
- `BoxDomain.outward_normal`
- `TensorProductProvider.box`

**Tests created/changed:**
- `tests/spaces/test_sobolev.py`
- `tests/operators/test_bessel.py`

**Review Status:** APPROVED with minor recommendations

**Git Commit Message:**
feat(pygeoinf3d): refactor box Sobolev phase to continuous-first API

- Replace public Sobolev3D and BesselOperator3D with Function3D-based APIs
- Keep spectral coefficient spaces explicit as secondary Spectral* classes
- Add kernel-backed Sobolev duals, contract checks, and 378 passing tests

Plan: pygeoinf3D/docs/agent-docs/active-plans/pygeoinf3d-foundation-plan.md
Phase: 4 of 7
Related: pygeoinf3D/docs/agent-docs/active-plans/pygeoinf3d-foundation-plan-phase-4-complete.md