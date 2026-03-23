# Foundational Demonstrations

This directory contains geometry-agnostic demonstrations of core pygeoinf3D abstractions: domains, spaces, functions, operators, and basis-independent representations.

These notebooks are suitable for a first-time introduction to the library and provide the mathematical foundations needed for geometry-specific demos.

## Notebooks (in recommended order)

1. **1_box_domain_intro.ipynb**
   - Introduction to 3D rectangular domains
   - BoxDomain construction and properties
   - Coordinate systems and axis-aligned bounding boxes
   - **Duration:** ~10 minutes

2. **2_functions_3d.ipynb**
   - Function3D objects: callable wrappers over 3D domains
   - Evaluation at single points and batches
   - Function composition and operations
   - **Duration:** ~15 minutes

3. **3_lebesgue_space_3d.ipynb**
   - L² Hilbert spaces: Lebesgue3D
   - Inner products and norms
   - Basis-free representation of functions
   - **Duration:** ~20 minutes

4. **4_kernel_functionals_3d.ipynb**
   - Integral kernels and linear functionals
   - VolumeKernel3D for domain integrals
   - Kernel providers and lazy evaluation
   - **Duration:** ~20 minutes

5. **5_providers_3d.ipynb**
   - Function and kernel providers: deferred evaluation and caching
   - FunctionProvider3D for parametric families
   - KernelProvider3D for measurement operators
   - **Duration:** ~15 minutes

6. **6_laplacian_3d.ipynb**
   - The Laplacian operator on boxes
   - BoxLaplacian and InverseBoxLaplacian
   - Spectral eigendecomposition (eigenvalues and eigenfunctions)
   - **Duration:** ~25 minutes

7. **7_bessel_sobolev_3d.ipynb**
   - Sobolev spaces with smoothing weights
   - BesselSobolevBox: spectral weight operator
   - Inverse Sobolev norms and adjoint computations
   - **Duration:** ~25 minutes

## Total Time: ~2 hours

## Key Concepts

- **Hilbert Space:** Abstract infinite-dimensional function space with inner product
- **Basis-free:** Representations do not depend on a discrete basis; operations are mesh-independent
- **Spectral operator:** Diagonalized in eigenbasis; supports fast batch evaluation
- **Adjoint:** Dual operator mapping back to the original space (enables inverse problems)

## Next Steps

Once you've completed the foundational notebooks, choose a geometry-specific track:

- **[box/sola/](../box/sola/)** — SOLA operator with volumetric kernels (generic case)
- **[box/pli/](../box/pli/)** — Path-integral measurements (fast spectral path)
- **[box/sampling/](../box/sampling/)** — Gaussian random field sampling via KL expansion
