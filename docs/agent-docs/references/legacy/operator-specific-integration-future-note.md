# Operator-Specific Integration: Future Design Note

## Purpose

Record the current architectural state of `pygeoinf3D` around integration and projection, and capture a recommended future direction for operators such as `BoxLaplacian` that may need dedicated numerical infrastructure.

This note is intentionally forward-looking. It is not an implementation plan and does not imply immediate code changes.

## Current State

### What is already separated cleanly

- `Region3D` handles geometric semantics.
- `Domain3D` defines the computational integration hook `integrate_volume(...)`.
- `StructuredDomain3D` adds structured numerical primitives such as:
  - `structured_mesh(...)`
  - `volume_weights(...)`
  - `integrate_volume_on_mesh(...)`
- `Lebesgue3D` remains pairing-first and basis-free by default.
- `VolumeKernel3D` evaluates dual pairings through `domain.integrate_volume(...)`.

This means the package already distinguishes:
- geometry and domain membership,
- generic domain integration,
- function-space semantics,
- operator logic.

### Where the current coupling still exists

The package does **not** yet have a first-class abstraction for operator-specific numerical backends.

Two examples:

1. `VolumeKernel3D`
   - In `pygeoinf3d.spaces.forms.VolumeKernel3D`, the pairing is hard-wired to:
     `domain.integrate_volume(self._kernel.pointwise_mul(g))`
   - This is appropriate for generic volumetric dual pairings, but it means the pairing rule is domain-driven rather than operator-driven.

2. `BoxLaplacian`
   - In `pygeoinf3d.operators.laplacian_box.BoxLaplacian`, projection is computed by directly calling:
     - `box.structured_mesh(...)`
     - `box.volume_weights(...)`
   - This is already an operator-specific numerical path, but it is embedded directly in the operator rather than delegated to reusable integration/projection infrastructure.

## Assessment

### Is the package compatible with future modular integration?

Yes, **partially**.

The current abstractions are good enough to support future modularization because:
- domains already expose generic integration hooks,
- structured domains already expose reusable low-level numerical primitives,
- operators are already free to bypass `integrate_volume(...)` when they need specialized numerics.

### Is the package fully structured for operator-specific backends today?

No.

The missing abstraction is a reusable backend layer for operator-specific numerical work.

At the moment, the architecture supports:
- domain-specific integration,
- ad hoc operator-specific numerics,

but not yet:
- explicit, swappable operator-level integration/projection backends.

## Why this matters for `BoxLaplacian`

The current `BoxLaplacian` is a quadrature-based Galerkin projection:

1. sample `f` on a structured mesh,
2. evaluate each eigenfunction on that mesh,
3. compute coefficients via weighted dot products,
4. reconstruct `\sum_k \lambda_k c_k \phi_k`.

This is a valid generic spectral implementation, but it is not the best long-term route for the box geometry when the basis is separable and homogeneous boundary conditions are supported.

For box domains with standard Dirichlet/Neumann conditions, a dedicated transform-based implementation (DST/DCT-style) would typically be:
- faster,
- more accurate,
- more obviously self-adjoint in discrete form,
- better aligned with the `intervalinf` interval Laplacian design.

That future operator should still look continuous at the public API level, but internally it should be allowed to delegate to dedicated numerical infrastructure.

## Recommended Future Design

### Keep the current generic path

Preserve `Domain3D.integrate_volume(...)` as the default continuous-looking integration interface.

Use it for:
- generic kernel pairings,
- non-structured domains,
- prototype operators,
- fallback implementations.

### Add an operator/backend layer

Introduce a new abstraction family dedicated to operator-specific numerical work. Examples of acceptable shapes:

- `QuadratureBackend`
- `SpectralProjectionBackend`
- `StructuredBoxTransformBackend`
- `BoxOperatorBackend`

The exact naming is less important than the boundary.

The important rule is:

> Domains provide geometry and generic integration capability; operators choose and use numerical backends appropriate to their structure.

### Expected responsibilities

#### Domains
Own:
- region and geometry,
- bounding boxes,
- generic `integrate_volume(...)`,
- structured meshes and compatible weights.

Do **not** own:
- basis choice,
- spectral truncation policy,
- transform selection,
- Gram solves,
- operator-specific projection rules.

#### Providers
Own:
- basis/eigenfunction enumeration,
- eigenvalues,
- geometric metadata tied to the basis family.

Do **not** own:
- coefficient extraction algorithms,
- projection solvers,
- domain quadrature policy.

#### Operator-specific backends
Own:
- analysis/projection rules,
- transform-based acceleration,
- quadrature caching,
- coefficient-space scaling,
- reconstruction strategy,
- operator-specific discrete approximations.

#### Operators
Own:
- continuous public semantics,
- parameter validation,
- selection or injection of the numerical backend,
- packaging the backend result back into `Function3D` / `LinearOperator` form.

## Practical Target for a Future `BoxLaplacian`

A good future direction would be:

1. Keep the current quadrature-based `BoxLaplacian` logic as a fallback backend.
2. Add a transform-based backend for homogeneous Dirichlet/Neumann box problems.
3. Allow the operator to select or receive the backend explicitly.

For example, conceptually:

- `BoxLaplacian(..., backend="quadrature")`
- `BoxLaplacian(..., backend="separable-transform")`
- or `BoxLaplacian(..., backend=SomeBackendObject(...))`

The public operator remains continuous-looking; only the numerical realization changes.

## Relationship to Existing Representation Work

This direction is consistent with the package's existing representation/backend split for coefficient workflows.

The representation architecture already encodes an important principle:

> Spaces should not be polluted with backend-specific numerical logic.

The same idea should be extended to structured operators.

In particular, `BoxLaplacian` should not permanently own private projection machinery if that machinery will later be reused by:
- inverse Laplacians,
- Sobolev preconditioners,
- Helmholtz-like operators,
- separable filtering operators,
- box-domain covariance operators.

## Bottom Line

- The current package structure is close enough to support this future cleanly.
- The required next abstraction is **not** more work inside `BoxDomain`.
- The required next abstraction is a reusable operator/backend layer sitting between:
  - structured domains and basis providers,
  - and continuous operators such as `BoxLaplacian`.

That is the right place for dedicated integration, projection, and transform numerics.
