## Plan: Noisy Path-Form Inversion Notebook

Create a new pygeoinf3D demo notebook that mirrors the existing least-norm inversion demo but uses noisy path-integral measurements on a low-order Sobolev model space. The notebook will generate 10 random straight rays from boundary sources to receivers on the top face, build a path-form SOLA operator, add Gaussian data noise via a data covariance model, and recover only the noisy minimum-norm solution.

**Phases 4**
1. **Phase 1: Scaffold notebook and path geometry**
    - **Objective:** Create the notebook structure, imports, Sobolev model space, and random path geometry with correct current APIs.
    - **Files/Functions to Modify/Create:** `pygeoinf3D/demos/noisy_path_form_least_norm_inversion_demo.ipynb`; `pygeoinf3D/docs/agent-docs/references/living/pygeoinf3d-reference.md`
    - **Tests to Write:** Notebook execution checks for setup cells and geometry summary output.
    - **Steps:**
        1. Create the new notebook with title, overview, imports, and reproducibility setup.
        2. Build the unit-cube domain, Lebesgue base space, low-order Sobolev space, and spectral provider.
        3. Generate 10 random straight paths with sources sampled from all boundary faces except the top face and receivers sampled on the top face boundary.
        4. Add path visualization and printed geometry diagnostics.

2. **Phase 2: Build forward operator and noisy data model**
    - **Objective:** Define the true model, construct path-integral measurements, and generate noisy synthetic data with an explicit covariance model.
    - **Files/Functions to Modify/Create:** `pygeoinf3D/demos/noisy_path_form_least_norm_inversion_demo.ipynb`
    - **Tests to Write:** Notebook execution checks for path-form construction and clean/noisy data summaries.
    - **Steps:**
        1. Write the `Function3D` ground-truth model callable.
        2. Construct `PathIntegralFunctional(sobolev, path, n_quad=...)` for all paths and stack them in `SOLAOperator3D`.
        3. Create a Gaussian data error measure on the Euclidean data space and sample noisy data through `LinearForwardProblem.synthetic_data` or the equivalent explicit noise draw.
        4. Add clean-versus-noisy measurement visualization and report the chosen covariance / noise scale.

3. **Phase 3: Implement noisy least-norm inversion**
    - **Objective:** Recover the model from noisy path-form data using `LinearMinimumNormInversion` with a data covariance model.
    - **Files/Functions to Modify/Create:** `pygeoinf3D/demos/noisy_path_form_least_norm_inversion_demo.ipynb`
    - **Tests to Write:** Notebook execution checks for inversion completion, residual reporting, and chi-squared diagnostics.
    - **Steps:**
        1. Build `LinearForwardProblem(G, data_error_measure=...)`.
        2. Compute the noisy minimum-norm operator via `LinearMinimumNormInversion.minimum_norm_operator(...)`.
        3. Apply the inverse to the noisy data only and report data misfit / chi-squared-style diagnostics.
        4. Confirm the recovered object is a `Function3D` and summarize the role of the data covariance in damping the inversion.

4. **Phase 4: Visualize results and update package reference**
    - **Objective:** Complete the notebook with model comparisons, sensitivity discussion, and living-reference updates.
    - **Files/Functions to Modify/Create:** `pygeoinf3D/demos/noisy_path_form_least_norm_inversion_demo.ipynb`; `pygeoinf3D/docs/agent-docs/references/living/pygeoinf3d-reference.md`
    - **Tests to Write:** Notebook execution validation for final visualization cells.
    - **Steps:**
        1. Add slice-based comparison between the true model and the recovered noisy inversion.
        2. Add path and data visualizations that explain the acquisition geometry and noisy observations.
        3. Write concluding remarks about Sobolev-domain path forms, boundary geometry, and data-side damping.
        4. Update the living reference to document the new notebook and its workflow.

**Open Questions 4**
1. Notebook only? Answered: `.ipynb` only.
2. Noisy solve branch? Answered: use `LinearMinimumNormInversion` with a data covariance model to induce damping.
3. Source locations? Answered: sample from all boundary faces except the top face.
4. Baseline solve? Answered: include only the noisy inversion.