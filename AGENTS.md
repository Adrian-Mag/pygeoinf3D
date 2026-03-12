# Agent Configuration for pygeoinf3D

## Plan Directory
`docs/agent-docs/`

All agent-oriented materials live in this directory:
- **`docs/agent-docs/active-plans/`** — In-progress plans
- **`docs/agent-docs/completed-plans/`** — Finished projects (reference archive)
- **`docs/agent-docs/references/`** — Research reports, exploration findings, and living references
- **`docs/agent-docs/theory/`** — Theory documents and research papers

## Package Context
**pygeoinf3D** is a planned extension package for concrete 3D function spaces and operators built on top of `pygeoinf`.

### Intended Scope
- 3D geometric domains embedded in `EuclideanSpace(3)`
- Region semantics represented by `pygeoinf.subsets.Subset` objects or thin `Region3D` wrappers
- Domain objects that add computational geometry on top of those regions
- Optional structured-domain layer for geometries with strong numerical structure
- Lebesgue and Sobolev spaces of scalar functions on 3D domains
- Spectral and quadrature-based operators for box/parallelepiped and ball geometries first

### Architectural Rule
- `Region3D` is a thin wrapper over `pygeoinf.subsets.Subset` and handles pure geometry semantics
- `Domain3D` is **not** a `HilbertSpace`
- `Domain3D` owns an ambient `EuclideanSpace(3)` and a geometric region
- `StructuredDomain3D` is an optional intermediate layer for domains with natural grids, coordinate systems, or spectral structure
- Function spaces inherit from `pygeoinf.HilbertSpace` and are defined **on** a `Domain3D`

## Package Quick References
All files matching `docs/agent-docs/references/living/*-reference.md` are condensed package references and should be read before source exploration once they exist.

## Dependencies
- Core: numpy, scipy, matplotlib
- Optional: pyshtools and Cartopy for spherical-harmonic and mapping workflows
