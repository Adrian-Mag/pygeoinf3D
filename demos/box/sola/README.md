# SOLA — Volumetric Kernel Measurements

This directory contains a demonstration of the **SOLA** (Standard Orthonormal Linearized Adjoint) operator-based workflow for least-norm inversion with volumetric kernel measurements on 3D box domains.

## Background

The SOLA operator is a general-purpose measurement operator that accepts any linear form:
- **VolumeKernel3D** (smooth spatial kernels, e.g., Gaussian blobs)
- **PathIntegralFunctional** (line integrals along curves)
- **SumLinearForm** (mixed stacks of any of the above)

It does **not** assume any specific basis or spectral structure, making it the go-to choice when you need a flexible, general-purpose forward operator.

## Notebooks

### sola_volumetric_kernels.ipynb

Demonstrates least-norm inversion using the SOLA operator with smooth Gaussian volumetric kernels.

**Topics covered:**
- Building a `SOLAOperator3D` on a `Lebesgue3D` space
- Computing the adjoint via `Sobolev3D.from_dual`
- Using pygeoinf's generic `LinearMinimumNormInversion` solver
- Recovering a model from noisy or synthetic data
- Visualization of kernels and solutions

**Preconditions:**
- Familiarity with `Lebesgue3D` (see `../foundational/3_lebesgue_space_3d.ipynb`)
- Understanding of kernel functionals (see `../foundational/4_kernel_functionals_3d.ipynb`)

**Output:**
- Solution Function3D (basis-free)
- Residual norm ||G(u) - d||

---

## Performance Characteristics

| Measurement Type | Space | Solver | Per-Call Cost | Pre-Computation |
|---|---|---|---|---|
| Volume kernels | Lebesgue3D | SOLAOperator3D | O(m·N) | None |
| | | + LinearMinimumNormInversion | O(m³) eigendecomposition | Per inversion |

**m** = number of measurements, **N** = basis/quadrature cost per kernel evaluation

---

## When to Use SOLA

✅ **Use SOLA when:**
- Measurements are smooth spatial kernels (not line integrals)
- You want a general-purpose, geometry-agnostic forward operator
- You don't have a spectral basis (or don't want to assume one)
- You plan to use pygeoinf's full inversion pipeline

❌ **Avoid SOLA when:**
- Measurements are path integrals AND you have a spectral weight → use **SpectralGramSolver** instead (1000–100,000× faster)
- You want minimal abstraction overhead → consider specialized bespoke solvers

---

## Next Steps

After this demo, consider:
- **[../pli/](../pli/)** — Fast path-integral inversion with spectral solvers (1000–100,000× speedup)
- **[../sampling/](../sampling/)** — Gaussian random field sampling via KL expansion
- **[../../docs/SOLVER_SELECTION_GUIDE.md](../../docs/SOLVER_SELECTION_GUIDE.md)** — Automatic solver selection via SolverRouter
