from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray

class PyNumBin:
    bin_id: int
    range: Tuple[float, float]
    count: int
    bin_pct: float
    pos: int
    neg: int
    woe: float
    iv: float
    event_rate: float
    is_missing: bool

class PyCatBin:
    bin_id: int
    indices: List[int]
    count: int
    bin_pct: float
    pos: int
    neg: int
    woe: float
    iv: float
    event_rate: float
    is_missing: bool

class NumericalBinning:
    def __init__(self, max_bins: int, min_bin_pct: float): ...
    def fit(self, x: NDArray[np.float64], y: NDArray[np.int32]) -> List[PyNumBin]: ...
    def transform(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...
    def fit_transform(
        self, x: NDArray[np.float64], y: NDArray[np.int32]
    ) -> NDArray[np.float64]: ...
    @property
    def bins(self) -> List[PyNumBin]: ...

class CategoricalBinning:
    def __init__(self, max_bins: int, min_bin_pct: float): ...
    def fit(self, x: NDArray[np.int32], y: NDArray[np.int32]) -> List[PyCatBin]: ...
    def transform(self, x: NDArray[np.int32]) -> NDArray[np.float64]: ...
    def fit_transform(
        self, x: NDArray[np.int32], y: NDArray[np.int32]
    ) -> NDArray[np.float64]: ...
    @property
    def bins(self) -> List[PyCatBin]: ...
