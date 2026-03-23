"""Volumetric kernel interaction backend for structured quadrature meshes.

Provides :class:`VolumeInteractionBackend`, which assembles and caches the
kernel-evaluation matrix

.. math::

    K_{ij} = f_i(\\mathbf{x}_j), \\quad i = 0, \\ldots, n_{\\text{kernels}}-1,
    \\quad j = 0, \\ldots, n_{\\text{pts}}-1,

and derives the Gram and cross-Gram matrices via weighted summation over the
quadrature points.

Design notes
------------
* Each instance owns its own quadrature points and weights — no shared state.
* ``K`` is computed lazily on the first call to :meth:`interaction_matrix`
  and cached thereafter.
* Kernel evaluations are dispatched to ``provider.evaluate_all(pts)``; the
  default loop-based implementation in :class:`FunctionProvider3D` is used
  when no vectorized override is present.
* :meth:`cross_gram` validates that both backends share the same quadrature
  mesh (identity check on the arrays) before computing the cross-product.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np

from pygeoinf3d.providers.base import FunctionProvider3D


class VolumeInteractionBackend:
    r"""Assembles and caches volumetric kernel evaluations on a structured
    quadrature mesh.

    Given a domain and a :class:`~pygeoinf3d.providers.base.FunctionProvider3D`
    with *n* functions, this backend assembles:

    * **interaction matrix** :math:`K \in \mathbb{R}^{n \times N}` where
      :math:`K_{ij} = f_i(\mathbf{x}_j)` and :math:`\mathbf{x}_j` are the
      quadrature points;
    * **Gram matrix** :math:`G \in \mathbb{R}^{n \times n}` where
      :math:`G_{ij} = \sum_k K_{ik}\,K_{jk}\,w_k \approx \int f_i f_j\,dV`;
    * **cross-Gram matrix** between two backends sharing the same quadrature
      mesh.

    The interaction matrix is computed once and cached; subsequent calls to
    :meth:`interaction_matrix`, :meth:`gram`, or :meth:`cross_gram` reuse the
    cached result at no extra cost.

    Parameters
    ----------
    domain:
        A :class:`~pygeoinf3d.geometries.box.domain.BoxDomain` (or any domain
        implementing :meth:`structured_mesh` and :meth:`volume_weights`).
    provider:
        A :class:`~pygeoinf3d.providers.base.FunctionProvider3D` whose
        functions are to be evaluated.
    resolution:
        Quadrature resolution — either a single integer *n* (giving an
        ``n×n×n`` grid) or a 3-tuple ``(nx, ny, nz)``.
    location:
        ``'cell_centers'`` (default) or ``'nodes'``.  Passed directly to
        :meth:`structured_mesh` and :meth:`volume_weights`.

    Examples
    --------
    ::

        import numpy as np
        from pygeoinf3d import BoxDomain
        from pygeoinf3d.providers.function_providers import GaussianProvider3D
        from pygeoinf3d.solvers import VolumeInteractionBackend

        domain   = BoxDomain(np.zeros(3), np.ones(3))
        centers  = np.array([[0.25, 0.25, 0.25], [0.75, 0.75, 0.75]])
        provider = GaussianProvider3D(domain, centers, sigmas=0.1)
        backend  = VolumeInteractionBackend(domain, provider, resolution=(16, 16, 16))

        K = backend.interaction_matrix()   # shape (2, 4096)
        G = backend.gram()                 # shape (2, 2)
    """

    def __init__(
        self,
        domain,
        provider: FunctionProvider3D,
        resolution: Union[int, Tuple[int, int, int]],
        location: str = "cell_centers",
    ) -> None:
        self._domain = domain
        self._provider = provider
        self._resolution = resolution
        self._location = location

        # Quadrature points and weights — owned by this instance; the domain's
        # internal _quad_cache means repeated calls here are essentially free.
        self._pts = domain.structured_mesh(resolution, location=location)
        self._weights = domain.volume_weights(resolution, location=location)

        # Interaction matrix cache — populated lazily on first access.
        self._K_cache: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def interaction_matrix(self) -> np.ndarray:
        """Evaluate all kernels at every quadrature point.

        Returns
        -------
        ndarray of shape ``(n_kernels, n_pts)``
            :math:`K_{ij} = f_i(\\mathbf{x}_j)`.  Computed once and cached.
        """
        if self._K_cache is None:
            self._K_cache = self._provider.evaluate_all(self._pts)
        return self._K_cache

    def gram(self) -> np.ndarray:
        r"""Compute the Gram matrix :math:`G = K\,\mathrm{diag}(w)\,K^\top`.

        .. math::

            G_{ij} = \sum_k K_{ik}\,w_k\,K_{jk}
                   \approx \int_\Omega f_i(\mathbf{x})\,f_j(\mathbf{x})\,\mathrm{d}V

        Returns
        -------
        ndarray of shape ``(n_kernels, n_kernels)``
        """
        K = self.interaction_matrix()
        w = self._weights
        # K @ diag(w) @ K.T  — written as matrix multiply to stay O(n²·N)
        return K @ (w[:, np.newaxis] * K.T)

    def cross_gram(self, other: "VolumeInteractionBackend") -> np.ndarray:
        r"""Compute the cross-Gram matrix between *self* and *other*.

        .. math::

            G_{ij} = \sum_k K^{(1)}_{ik}\,w_k\,K^{(2)}_{jk}
                   \approx \int_\Omega f_i(\mathbf{x})\,g_j(\mathbf{x})\,\mathrm{d}V

        Both backends must share the same quadrature mesh (same domain,
        resolution, and location).

        Parameters
        ----------
        other:
            Another :class:`VolumeInteractionBackend` built with the same
            domain, resolution, and location.

        Returns
        -------
        ndarray of shape ``(n_kernels_self, n_kernels_other)``

        Raises
        ------
        ValueError
            If the two backends do not share the same quadrature points.
        """
        if (
            self._domain is not other._domain
            or self._resolution != other._resolution
            or self._location != other._location
        ):
            raise ValueError(
                "cross_gram requires both backends to share the same "
                "domain, resolution, and location.  "
                f"self: resolution={self._resolution}, location={self._location!r}; "
                f"other: resolution={other._resolution}, location={other._location!r}."
            )
        K_self = self.interaction_matrix()
        K_other = other.interaction_matrix()
        w = self._weights
        return K_self @ (w[:, np.newaxis] * K_other.T)
