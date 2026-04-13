from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Greeks:
    """
    Class for storing values of Greeks, data container.
    Calculated risk values are immutable because of frozen parameter - once the Greeks are calculated and stored, they can never be changed; this is added for safety.
    """
    delta: float | np.ndarray
    gamma: float | np.ndarray
    vega: float | np.ndarray
    theta: float | np.ndarray
    rho: float | np.ndarray

