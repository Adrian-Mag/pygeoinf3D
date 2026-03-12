# pygeoinf3D

**pygeoinf3D** is a planned extension package providing concrete 3D function spaces, geometric domains, and differential operators built on top of [`pygeoinf`](../pygeoinf).

## Architecture Overview

The package follows a strict three-layer separation:

```
pygeoinf.Subset / EuclideanSpace(3)   ← semantic geometry (pygeoinf)
    └── Region3D                       ← thin geometric wrapper (pygeoinf3d.core.region)
            └── Domain3D               ← computational domain (pygeoinf3d.core.domain_base)
                    └── StructuredDomain3D   ← domains with natural grids/coords
```

Function spaces (e.g. `Lebesgue3D`, `Sobolev3D`) will inherit from `pygeoinf.HilbertSpace`
and be **defined on** a `Domain3D` — they are *not* domains themselves.

## Dependencies

| Group       | Packages                     | Status     |
|-------------|------------------------------|------------|
| Core        | numpy, scipy, matplotlib     | Required   |
| Geometry    | trimesh, meshzoo             | Optional   |
| Spherical   | pyshtools, Cartopy           | Optional   |
| Heavy FEM   | FEniCS, dolfinx, ngsolve     | Deferred (v2+) |

## Planned Phases

1. **Phase 1** — Package skeleton, `Region3D`, `Domain3D`, `StructuredDomain3D`, `Boundary3D`
2. **Phase 2** — `Function3D` representation and `Lebesgue3D` Hilbert space
3. **Phase 3** — `BoxDomain` (tensor-product, first geometry MVP)
4. **Phase 4** — Sobolev/Bessel operators on box domains
5. **Phase 5** — `BallDomain` (spherical coordinate support)
6. **Phase 6** — First end-to-end inversion workflow

## Quick Start (Phase 1 skeleton)

```python
import numpy as np
from pygeoinf import EuclideanSpace, Ball
from pygeoinf3d.core.region import Region3D
from pygeoinf3d.core.config import AxisAlignedBoundingBox

space = EuclideanSpace(3)
ball  = Ball(space, np.zeros(3), 1.0)
bbox  = AxisAlignedBoundingBox(low=np.full(3, -1.0), high=np.full(3, 1.0))
region = Region3D(ball, bounding_box=bbox)

print(region.ambient_space)       # EuclideanSpace(3)
print(region.contains(np.zeros(3)))  # True
```
# pygeoinf3D
