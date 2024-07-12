import numpy as np
from dataclasses import dataclass
from typing import Optional, Protocol

@dataclass
class Data:
    """
    Data class for storing input data and distance matrix.
    (X, D, label)
    """
    X: Optional[np.ndarray] # [n, m] where n is the number of data points and m is the number of features
    D: np.ndarray # [n, 2] where n is the number of data points
    Label: Optional[np.ndarray] # [n] where n is the number of data points