from dataclasses import dataclass

@dataclass(frozen=True)
class Greeks:
    """
    Class for storing values of Greeks, data container.
    Calculated risk values are immutable because of frozen - once the Greeks are calculated and stored, they can never be changed; this is added for safety.
    """
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float

