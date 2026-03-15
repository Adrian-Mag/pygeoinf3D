## Phase 2 Complete: Function Representation and 3D Lebesgue Space

Implemented the first primal function-space layer for `pygeoinf3D` under the pairing-first conventions inherited from `pygeoinf`. This phase added a minimal primal-only `Function3D`, a basis-free `Lebesgue3D` built on `Domain3D.integrate_volume`, kernel-backed dual representers via `VolumeKernel3D`, updated living documentation, and a fully passing Phase 2 test suite.

**Files created/changed:**
- pygeoinf3D/pygeoinf3d/__init__.py
- pygeoinf3D/pygeoinf3d/core/__init__.py
- pygeoinf3D/pygeoinf3d/core/functions.py
- pygeoinf3D/pygeoinf3d/spaces/__init__.py
- pygeoinf3D/pygeoinf3d/spaces/forms.py
- pygeoinf3D/pygeoinf3d/spaces/lebesgue.py
- pygeoinf3D/tests/core/test_functions.py
- pygeoinf3D/tests/spaces/__init__.py
- pygeoinf3D/tests/spaces/test_lebesgue.py
- pygeoinf3D/docs/agent-docs/active-plans/pygeoinf3d-foundation-plan.md
- pygeoinf3D/docs/agent-docs/references/living/pygeoinf3d-reference.md

**Functions created/changed:**
- `Function3D.__init__`
- `Function3D.domain`
- `Function3D.space`
- `Function3D.is_attached`
- `Function3D.__call__`
- `Function3D.__add__`
- `Function3D.__sub__`
- `Function3D.__neg__`
- `Function3D.__mul__`
- `Function3D.__rmul__`
- `Function3D.__truediv__`
- `Function3D.pointwise_mul`
- `Function3D.copy`
- `Function3D.attach_to_space`
- `Function3D.detach`
- `Lebesgue3D.__init__`
- `Lebesgue3D.dim`
- `Lebesgue3D.function_domain`
- `Lebesgue3D.to_dual`
- `Lebesgue3D.from_dual`
- `Lebesgue3D.to_components`
- `Lebesgue3D.from_components`
- `Lebesgue3D.__eq__`
- `Lebesgue3D.__hash__`
- `Lebesgue3D.zero`
- `Lebesgue3D.copy`
- `Lebesgue3D.add`
- `Lebesgue3D.subtract`
- `Lebesgue3D.multiply`
- `Lebesgue3D.negative`
- `Lebesgue3D.inner_product`
- `Lebesgue3D.norm`
- `VolumeKernel3D.__init__`
- `VolumeKernel3D.components`
- `VolumeKernel3D._mapping_impl`
- `VolumeKernel3D.copy`
- `VolumeKernel3D.__neg__`
- `VolumeKernel3D.__mul__`
- `VolumeKernel3D.__rmul__`
- `VolumeKernel3D.__truediv__`
- `VolumeKernel3D.__add__`
- `VolumeKernel3D.__sub__`
- `VolumeKernel3D.__imul__`
- `VolumeKernel3D.__iadd__`
- `VolumeKernel3D.__isub__`
- `VolumeKernel3D.kernel`

**Tests created/changed:**
- `pygeoinf3D/tests/core/test_functions.py`
- `pygeoinf3D/tests/spaces/test_lebesgue.py`
- Full `pygeoinf3D` test suite: 141 tests passing

**Review Status:** APPROVED with minor recommendations

**Git Commit Message:**
feat(spaces): add pairing-first Function3D and Lebesgue3D

- Add primal-only Function3D and basis-free Lebesgue3D on Domain3D
- Implement kernel-backed dual representers via VolumeKernel3D with safe algebra
- Add Phase 2 tests and update living reference with 141 passing tests

Plan: pygeoinf3D/docs/agent-docs/active-plans/pygeoinf3d-foundation-plan.md
Phase: 2 of 7
Related: pygeoinf3D/docs/agent-docs/active-plans/pygeoinf3d-foundation-plan-phase-2-complete.md