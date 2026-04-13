import numpy as np
import warnings

from core.pricer_base import BasePricer
from core.option_base import BaseOption
from instruments.european import EuropeanOption
from instruments.asian import AsianOption
from market.environment import MarketEnvironment

class MonteCarloPricer(BasePricer):
    """Option pricing using Monte Carlo simulation. Assumes that the underlying asset price follows Geometric Brownian Motion using the exact GBM scheme (not Euler-Maruyama). For European options, the terminal price is simulated directly in a single step because these options are path-independent. The num_steps argument is unused in this case. For path-dependent options (e.g. Asian options), the full price path is simulated. The path has shape (num_paths, num_steps + 1), where the first column is the initial spot price S_0 and each subsequent column represents the asset price after one time step. The average price used in Asian option payoffs includes S_0. Here, num_steps refers to the number of stochastic time steps (discrete time intervals), consistent with the convention that a path with num_steps steps has num_steps + 1 observation points.

    Args:
        num_paths: Number of simulated paths (samples).
        num_steps: Number of time steps per path. Defaults to 1. Unused for
            European options. For Asian options, controls the number of
            observation points in the averaging (num_steps + 1 including S_0).
        random_seed: Seed for the random number generator. If None, results
            are not reproducible. If specified, must be a non-negative integer.
    """

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
        # Vectorized pricing is only supported by the BlackScholesMertonPricer - check if market_env variables are not arrays
        vars = {
            "spot_price": market_env.spot_price,
            "risk_free_rate": market_env.risk_free_rate,
            "volatility": market_env.volatility,
            "dividend_yield": market_env.dividend_yield
        }
        arrays = [name for name, value in vars.items() if isinstance(value, np.ndarray)]
        if arrays:
            np_arrays = ", ".join(arrays)
            raise TypeError(f"The MonteCarloPricer currently only supports scalar (single number) market inputs, while these inputs are NumPy arrays: {np_arrays}. Vectorized pricing is only supported by the BlackScholesMertonPricer.")
        
        # Value the option
        T_minus_t = option.time_to_maturity(market_env.pricing_date)

        if isinstance(option, EuropeanOption):
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
        elif isinstance(option, AsianOption):
            if T_minus_t < self.TOLERANCE: # Option value at expiry - meaningless for Asian options (need the asset price path to compute the payoff - path-dependent option)
                warnings.warn("Asian option pricing at maturity (T=0) is meaningless without historical price path data. The average price cannot be computed. Returning 0.0.")
                return 0.0
            else: # full path simulation
                rng = np.random.default_rng(self.random_seed) # set random seed
                Z_array = rng.standard_normal(size=(self.num_paths, self.num_steps))
                dt = T_minus_t/self.num_steps
                exp_array = np.exp((market_env.risk_free_rate - market_env.dividend_yield - 0.5 * market_env.volatility**2) * dt + market_env.volatility * np.sqrt(dt) * Z_array) # using the exact scheme applied step by step

                price_paths = market_env.spot_price * np.cumprod(exp_array, axis=1) # cumulative product for each path, computes the array of S_t of shape (num_paths, num_steps)
                S_0_col = np.full(shape=(self.num_paths, 1), fill_value=market_env.spot_price)
                price_paths = np.concatenate((S_0_col, price_paths), axis=1) # (num_paths, num_steps + 1) shape where the first column is a column of spot prices

                payoffs = option.get_payoff(price_paths) # Asian option payoffs for each simulated path, 1D array of shape (num_paths,)

                # Expected discounted payoff = option value
                option_value = np.exp(-market_env.risk_free_rate * T_minus_t) * np.mean(payoffs)

                return float(option_value)
        else:
            raise TypeError("Monte Carlo pricing is currently implemented only for EuropeanOption or AsianOption instances!")    


