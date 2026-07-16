from abc import ABC, abstractmethod
from typing import List, Tuple


class Lattice(ABC):
    """
    Abstract base class representing a physical lattice structure.
    """

    @property
    @abstractmethod
    def n_sites(self) -> int:
        """Return the total number of sites in the lattice."""
        pass

    @abstractmethod
    def get_edges(self, pbc: bool = False) -> list[tuple[int, int]]:
        """
        Return the list of coupling edges (bonds) between adjacent sites.

        Args:
            pbc: Enable Periodic Boundary Conditions (PBC).

        Returns:
            List of tuples of (site_idx_1, site_idx_2) representing bonds.
        """
        pass


class ChainLattice(Lattice):
    """
    A 1D chain of sites.
    """

    def __init__(self, n_sites: int):
        if n_sites < 2:
            raise ValueError("A chain lattice must have at least 2 sites.")
        self._n_sites = n_sites

    @property
    def n_sites(self) -> int:
        return self._n_sites

    def get_edges(self, pbc: bool = False) -> list[tuple[int, int]]:
        edges = []
        for i in range(self._n_sites - 1):
            edges.append((i, i + 1))
        if pbc:
            edges.append((self._n_sites - 1, 0))
        return edges


class SquareLattice(Lattice):
    """
    A 2D square grid lattice.
    """

    def __init__(self, width: int, height: int):
        if width < 1 or height < 1:
            raise ValueError("Lattice dimensions must be at least 1x1.")
        if width * height < 2:
            raise ValueError("Square lattice must have at least 2 sites total.")
        self.width = width
        self.height = height

    @property
    def n_sites(self) -> int:
        return self.width * self.height

    def _coord_to_idx(self, x: int, y: int) -> int:
        return y * self.width + x

    def get_edges(self, pbc: bool = False) -> list[tuple[int, int]]:
        edges = []

        # 1. Horizontal couplings
        for y in range(self.height):
            for x in range(self.width):
                idx = self._coord_to_idx(x, y)
                if x < self.width - 1:
                    next_idx = self._coord_to_idx(x + 1, y)
                    edges.append((idx, next_idx))
                elif pbc and self.width > 1:
                    next_idx = self._coord_to_idx(0, y)
                    edges.append((idx, next_idx))

        # 2. Vertical couplings
        for x in range(self.width):
            for y in range(self.height):
                idx = self._coord_to_idx(x, y)
                if y < self.height - 1:
                    next_idx = self._coord_to_idx(x, y + 1)
                    edges.append((idx, next_idx))
                elif pbc and self.height > 1:
                    next_idx = self._coord_to_idx(x, 0)
                    edges.append((idx, next_idx))

        return edges
