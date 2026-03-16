## Phase 2 Complete: Re-anchor Sobolev3D Dual Space

`Sobolev3D.to_dual` now returns a `VolumeKernel3D` whose `.domain` is the `Sobolev3D`
itself (not `base_space`). All three `from_dual` branches enforce strict
`fp.domain is not self` — the Phase 1 compatibility bridge is gone.
`VolumeKernel3D` arithmetic is generalised (no more `cast("Lebesgue3D", …)`,
error messages say "different space instances"). 88/88 sobolev+path tests pass;
6 SOLA failures remain as planned Phase 3 targets.

**Files created/changed:**
- `pygeoinf3d/spaces/sobolev.py`
- `pygeoinf3d/spaces/forms.py`
- `tests/spaces/test_sobolev.py`

**Functions created/changed:**
- `Sobolev3D.to_dual` — returns `VolumeKernel3D(self, Wf)` not `(base_space, Wf)`
- `Sobolev3D.from_dual` Branch 1 — strict `fp.domain is not self`
- `Sobolev3D.from_dual` Branch 2 — bridge collapsed to `fp.domain is not self`
- `Sobolev3D.from_dual` Branch 3 — bridge collapsed to `fp.domain is not self`
- `VolumeKernel3D.copy` / `__neg__` / `__mul__` — removed `cast("Lebesgue3D", …)`
- `VolumeKernel3D.__add__` / `__sub__` / `__iadd__` / `__isub__` — generalised error messages

**Tests created/changed:**
- `TestSobolev3DToDual.test_to_dual_domain_is_sobolev` (renamed + assertion fixed)
- `TestSobolev3DFromDual.test_rejects_volume_kernel_from_wrong_space` (match=`"different space instance"`)

**Review Status:** APPROVED

**Git Commit Message:**
```
refactor(spaces): re-anchor Sobolev3D dual to self, not base_space

- Sobolev3D.to_dual: VolumeKernel3D(self, Wf) instead of (base_space, Wf)
- Sobolev3D.from_dual all three branches: strict fp.domain is not self
- VolumeKernel3D arithmetic: remove cast("Lebesgue3D"), generalise error messages
- test_sobolev.py: rename test + fix assertion to fp.domain is sobolev

Plan: pygeoinf3D/docs/agent-docs/active-plans/sobolev-path-form-domain-refactor-plan.md
Phase: 2 of 4
Related: pygeoinf3D/docs/agent-docs/active-plans/sobolev-path-form-domain-refactor-phase-2-complete.md
```
