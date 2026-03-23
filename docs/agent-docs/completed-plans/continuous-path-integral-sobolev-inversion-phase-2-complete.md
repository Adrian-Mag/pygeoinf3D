## Phase 2 Complete: General Path Geometry Input Layer

Added `Path3D` to `pygeoinf3D` — a unified parameterized-curve abstraction supporting smooth callable parameterizations, polylines, and ordered point samples (piecewise-linear). All modes expose a normalized `t ∈ [0, 1]` arc-length interface and an `arclength` property. The degeneracy guard for zero-length callable skeletons was added during review.

**Files created/changed:**
- pygeoinf3D/pygeoinf3d/core/paths.py
- pygeoinf3D/tests/core/test_paths.py
- pygeoinf3D/pygeoinf3d/core/__init__.py
- pygeoinf3D/pygeoinf3d/__init__.py
- pygeoinf3D/docs/agent-docs/references/living/pygeoinf3d-reference.md

**Functions created/changed:**
- `Path3D.from_callable`
- `Path3D.from_polyline`
- `Path3D.from_points`
- `Path3D.__call__` (evaluation at normalized `t`)
- `Path3D.arclength`
- `Path3D.arc_length_nodes`
- `Path3D.domain`
- `Path3D.__repr__`
- internal helpers: `_chord_lengths`, `_approx_arclength_callable`, `_build_callable_evaluator`, `_build_polyline_evaluator`

**Tests created/changed:**
- `tests/core/test_paths.py::TestCallablePath` — evaluation shape, arclength accuracy, arc-length nodes
- `tests/core/test_paths.py::TestPolylinePath` — endpoint/midpoint evaluation, exact arclength, node count
- `tests/core/test_paths.py::TestPointSamplePath` — equivalence to polyline, endpoint agreement
- `tests/core/test_paths.py::TestPathDomain` — domain stored and None-default
- `tests/core/test_paths.py::TestPathRepr` — mode name and arclength appear in repr

**Review Status:** APPROVED (after adding callable degeneracy guard and lint fixes)

**Git Commit Message:**
feat(core): add Path3D geometry input layer

- support smooth callable, polyline, and point-sample path construction
- unified normalized arc-length interface: path(t), t in [0, 1]
- add degenerate-path guard for zero-length callable skeletons
- update exports, all 25 tests pass

Plan: pygeoinf3D/docs/agent-docs/active-plans/continuous-path-integral-sobolev-inversion-plan.md
Phase: 2 of 5
Related: pygeoinf3D/docs/agent-docs/active-plans/continuous-path-integral-sobolev-inversion-phase-2-complete.md
