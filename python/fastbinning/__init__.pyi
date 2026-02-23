from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray

class PyNumBin:
    """Result container for a single numerical bin."""

    bin_id: int
    range: Tuple[float, float]
    pos: int
    neg: int
    woe: float
    iv: float
    is_missing: bool

class PyCatBin:
    """Result container for a single categorical bin."""

    bin_id: int
    indices: List[int]
    pos: int
    neg: int
    woe: float
    iv: float
    is_missing: bool

class NumericalBinning:
    """High-performance numerical binning using Dynamic Programming
    with Monotonic WoE constraint."""
    def __init__(self, max_bins: int, min_bin_pct: float):
        """Args:
        max_bins: Maximum number of bins to create.
        min_bin_pct: Minimum proportion of samples required in each bin (0.0 to 1.0).
        """
        ...

    def fit(self, x: NDArray[np.float64], y: NDArray[np.int32]) -> List[PyNumBin]:
        """Fits the binning model to numerical data.

        Args:
            x: 1D NumPy array of numerical features (supports NaNs).
            y: 1D NumPy array of binary targets (0 or 1).
        Returns:
            A list of PyNumBin objects representing the optimal bins.
        """
        ...

    def transform(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Transforms the numerical data to fitted woe.

        Args:
            x: 1D NumPy array of numerical features (supports NaNs).

        Returns:
            1D NumPy array of woe values.
        """
        ...

class CategoricalBinning:
    """High-performance categorical binning with Monotonic WoE constraint."""
    def __init__(self, max_bins: int, min_bin_pct: float):
        """Args:
        max_bins: Maximum number of bins to create.
        min_bin_pct: Minimum proportion of samples required in each bin (0.0 to 1.0).
        """
        ...

    def fit(self, x: NDArray[np.int32], y: NDArray[np.int32]) -> List[PyCatBin]:
        """Fits the binning model to categorical data.

        Args:
            x: 1D NumPy array of encoded category indices.
            y: 1D NumPy array of binary targets (0 or 1).

        Returns:
            A list of PyCatBin objects.
        """
        ...

    def transform(self, x: NDArray[np.int32]) -> NDArray[np.float64]:
        """Transforms the categorical data to fitted woe.

        Args:
            x: 1D NumPy array of coded features (treat NaNs as -1).

        Returns:
            1D NumPy array of woe values.
        """
        ...
