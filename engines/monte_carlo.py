import numpy as np

from core.pricer_base import BasePricer
from core.option_base import BaseOption
from instruments.european import EuropeanOption
from market.environment import MarketEnvironment

class MonteCarloPricer(BasePricer):
    """Option pricing using Monte Carlo simulation. Assumes that the asset path follows Geometric Brownian Motion. Please note that the argument num_steps is unused for European options (terminal price is simulated directly) because European options are path-independent, but it is required for path-dependent exotics (e.g. Asian options) where the full price path must be simulated."""

    TOLERANCE = 1e-10 # class-level constant for floating-point comparisons

    def __init__(self, num_paths: int, num_steps: int = 1, random_seed: int | None = None):
        self.num_paths = num_paths
        self.num_steps = num_steps 
        self.random_seed = random_seed

        # Input data validation
        self._validate_num_paths()
        self._validate_num_steps()
        self._validate_random_seed()

    # Data validation methods:  
    def _validate_num_paths(self):
        if not isinstance(self.num_paths, int):
            raise TypeError(f"num_paths must be an integer (type int), got {type(self.num_paths)}")
        if self.num_paths < 1:
            raise ValueError("num_paths must be at least 1")  
        
    def _validate_num_steps(self):
        if not isinstance(self.num_steps, int):
            raise TypeError(f"num_steps must be an integer (type int), got {type(self.num_steps)}")  
        if self.num_steps < 1:
            raise ValueError("num_steps must be at least 1")
        
    def _validate_random_seed(self):
        if self.random_seed is not None:
            if not isinstance(self.random_seed, int):
                raise TypeError(f"If a random_seed is specified, it must be an integer (type int), got {type(self.random_seed)}") 
            elif self.random_seed < 0:
                raise ValueError("If a random_seed is specified, it cannot be negative")
        
    # The core of this class - pricing method:
    def _calculate_price(self, option: BaseOption, market_env: MarketEnvironment) -> float:
        if not isinstance(option, EuropeanOption):
            raise TypeError("Monte Carlo pricing is currently implemented only for EuropeanOption instances!")    
        
        T_minus_t = option.time_to_maturity(market_env.pricing_date)

        if T_minus_t < self.TOLERANCE: # Get option value at expiry
            return float(option.get_payoff(market_env.spot_price))
        else:
            rng = np.random.default_rng(self.random_seed) # set random seed

            # Terminal asset prices - ndarray
            S_terminal_array = market_env.spot_price * np.exp((market_env.risk_free_rate - market_env.dividend_yield - 0.5 * market_env.volatility**2) * T_minus_t + market_env.volatility * np.sqrt(T_minus_t) * rng.standard_normal(self.num_paths))

            # Terminal payoffs
            V_array = option.get_payoff(S_terminal_array)

            # Expected discounted payoff = option value for European options
            option_value = np.exp(-market_env.risk_free_rate * T_minus_t) * np.mean(V_array)

            return float(option_value)

