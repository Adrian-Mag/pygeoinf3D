# pygeoinf3D

**pygeoinf3D** is an extension package providing concrete 3D function spaces,
geometric domains, and measurement operators built on top of
[`pygeoinf`](../pygeoinf).  The current focus is **volumetric averaging kernels
on box domains** and the `SOLAOperator3D` measurement operator.

## Architecture Overview

The package follows a strict three-layer separation:

```
pygeoinf.Subset / EuclideanSpace(3)   ← semantic geometry (pygeoinf)
    └── Region3D                       ← thin geometric wrapper (pygeoinf3d.core.region)
            └── Domain3D               ← computational domain (pygeoinf3d.core.domain_base)
                    └── StructuredDomain3D   ← domains with natural grids/coords
                            └── BoxDomain    ← implemented (tensor-product box)
```

Function spaces (`Lebesgue3D`, …) inherit from `pygeoinf.HilbertSpace` and are
**defined on** a `Domain3D` — they are not domains themselves.

## Implemented Features

| Component | Class(es) | Notes |
|-----------|-----------|-------|
| Core geometry | `Region3D`, `Domain3D`, `StructuredDomain3D`, `Boundary3D` | Phase 1 |
| Functions | `Function3D`, `Lebesgue3D`, `VolumeKernel3D` | Phase 2 |
| Box geometry | `BoxDomain` | Phase 3, first concrete domain |
| Spectral basis | `TensorProductProvider` | Dirichlet/Neumann box eigenfunctions |
| Spectral operators | `SpectralPowerBase3D`, `GeneralizedBoxLaplacian` | Integer powers of (−α∇²)ˢ on BoxDomain |
| Spectral operator wrappers | `BoxLaplacian`, `InverseBoxLaplacian` | Thin wrappers over `GeneralizedBoxLaplacian` |
| Function providers | `GaussianProvider3D`, `BumpProvider3D`, `BoxcarProvider3D` | Volumetric kernels |
| Kernel adapter | `FunctionToKernelAdapter` | FunctionProvider3D → KernelProvider3D |
| Measurement operator | `SOLAOperator3D` | Forward, dual, and adjoint |
| Coefficient representations | `LebesgueRepresentation3D`, `OrthonormalProjectionBackend`, `GramProjectionBackend` | Finite-dim projection |
| Visualization | `evaluate_function3d_on_axis_slice`, `plot_function3d_axis_slice` | Slice plots |

## Dependencies

| Group       | Packages                     | Status     |
|-------------|------------------------------|------------|
| Core        | numpy, scipy, matplotlib     | Required   |
| Geometry    | trimesh, meshzoo             | Optional   |
| Spherical   | pyshtools, Cartopy           | Optional   |
| Heavy FEM   | FEniCS, dolfinx, ngsolve     | Deferred (v2+) |

## Volumetric-Only Boundary

All current kernel objects compute **3D volume integrals**.  Surface-integral,
line-integral, and point-evaluation kernels are not yet implemented.  Adding
non-volumetric kernels would require coordinated changes across the pipeline:

- `KernelProvider3D.get_kernel` currently has return type `VolumeKernel3D`; a
  new kernel family needs a compatible `LinearForm` subclass and updated type
  contracts.
- The sequence-backed `SOLAOperator3D` constructor enforces
  `isinstance(kernel, VolumeKernel3D)` and will reject other `LinearForm`
  subclasses outright.
- `Lebesgue3D.from_dual` only accepts `VolumeKernel3D`; the **adjoint mapping**
  requires a parallel extension here to support non-volumetric duals.

In short, the current pipeline is volume-kernel-centric by design; non-volumetric
kernels are not drop-in additions.

## Quick Start

```python
import numpy as np
from pygeoinf import EuclideanSpace
from pygeoinf3d import (
    BoxDomain, Lebesgue3D, Function3D,
    FunctionToKernelAdapter, SOLAOperator3D,
)
from pygeoinf3d.providers.function_providers import GaussianProvider3D
from pygeoinf3d.plot import plot_function3d_axis_slice

# 1. Domain and L² space
box   = BoxDomain(np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0]))
space = Lebesgue3D(box)

# 2. Gaussian volume-averaging kernels
centers = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
fprov   = GaussianProvider3D(box, centers=centers, sigmas=0.2)
kprov   = FunctionToKernelAdapter(fprov, space)

# 3. SOLA measurement operator  A : Lebesgue3D → R^m
op = SOLAOperator3D(space, EuclideanSpace(fprov.dim), kprov)

# 4. Apply to a test function  → measurement vector
#    Callable must handle (3,) single point → scalar and (n,3) batch → (n,)
def f_callable(x):
    x = np.asarray(x)
    return np.exp(-np.sum(x**2, axis=-1))

f    = Function3D(box, evaluate_callable=f_callable)
meas = op(f)   # shape (2,) — one volume integral per kernel

# 5. Inspect a kernel on an axis slice
plot_function3d_axis_slice(kprov.get_kernel(0).kernel, axis='z', coord=0.0)
```

## Spectral Operator Family

The generalized spectral operator family represents integer powers of the
positive box operator $(-\alpha\Delta)^s$:

```python
from pygeoinf3d import GeneralizedBoxLaplacian, TensorProductProvider, Lebesgue3D
from pygeoinf3d.geometries.box.domain import BoxDomain
import numpy as np

box      = BoxDomain(np.zeros(3), np.ones(3))
space    = Lebesgue3D(box)
provider = TensorProductProvider(box, truncation=(5, 5, 5), bc_type='dirichlet')

# Inverse Laplacian (covariance-like, for KL sampling)
inv_lap = GeneralizedBoxLaplacian(space, provider, power=-1)

# Square Laplacian (higher-order smoothing)
sq_lap  = GeneralizedBoxLaplacian(space, provider, power=2)

# Identity on Lebesgue3D
ident   = GeneralizedBoxLaplacian(space, provider, power=0)
```

**Convention:** powers are taken of the *positive* operator $(-\alpha\Delta)$,
not the raw $\Delta$.  This keeps all Neumann/Dirichlet eigenvalues
non-negative.  Only integer powers are supported in the current release.
For negative powers, modes satisfying $|\lambda_k| < \varepsilon$ are
suppressed to zero (pseudoinverse / mean-zero convention).

`BoxLaplacian` and `InverseBoxLaplacian` remain available as semantic
shorthand and are implemented as thin wrappers:
- `BoxLaplacian(space, provider)` ≡ `GeneralizedBoxLaplacian(..., power=1, alpha=1.0)`
- `InverseBoxLaplacian(space, provider)` ≡ `GeneralizedBoxLaplacian(..., power=-1)` for the default pseudoinverse mode

## Planned Next Steps

- Fractional / non-integer spectral powers via exact functional calculus
- Shifted/Bessel-Sobolev variants: $(−\alpha\Delta + \delta I)^s$
- `BallDomain` (spherical coordinate support)
- Surface-integral and point-evaluation kernel families
- End-to-end inversion workflow demo
