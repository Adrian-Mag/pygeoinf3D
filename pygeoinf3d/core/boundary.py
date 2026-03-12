"""Boundary3D: A lightweight first-class boundary object for 3D domains.

A `Boundary3D` represents the $(n-1)$-dimensional surface $\\partial\\Omega$ of
a `Domain3D`, providing membership testing and a link back to the
parent domain.
It has topological dimension 2 (a surface embedded in $\\mathbb{R}^3$).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from pygeoinf.hilbert_space import EuclideanSpace

if TYPE_CHECKING:
    from .domain_base import Domain3D


class Boundary3D(ABC):
    """Abstract first-class computational boundary tied to a parent `Domain3D`.

    `Boundary3D` captures the boundary surface $\\partial\\Omega$ of a 3D
    computational domain.  It deliberately stays lightweight: it provides
    membership testing, links back to the parent domain, and declares its
    topological dimension.  Quadrature, meshing, and normal computation are
    added by concrete subclasses (e.g. within `StructuredDomain3D` subclasses).

    Subclasses must implement:
        - `contains`

    Args:
        parent_domain: The `Domain3D` whose boundary this object represents.
    """

    def __init__(self, parent_domain: "Domain3D") -> None:
        self._parent_domain = parent_domain

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def parent_domain(self) -> "Domain3D":
        """The `Domain3D` this boundary belongs to."""
        return self._parent_domain

    @property
    def ambient_space(self) -> EuclideanSpace:
        """Returns the ambient `EuclideanSpace(3)` from the parent domain."""
        return self._parent_domain.ambient_space

    @property
    def dim(self) -> int:
        """Topological dimension of the boundary surface (always 2)."""
        return 2

    @abstractmethod
    def contains(self, point: np.ndarray) -> bool:
        """Test whether *point* lies on the boundary surface.

        Args:
            point: Array of shape ``(3,)``.

        Returns:
            ``True`` if the point is on the boundary.
        """
