import numpy as np
from scipy.stats import binom

from core.pricer_base import BasePricer
from core.option_base import BaseOption
from instruments.european import EuropeanOption
from instruments.american import AmericanOption
from market.environment import MarketEnvironment

class BinomialTreePricer(BasePricer):
    """Option pricing using Cox, Ross, and Rubinstein binomial model (1979). 
    Implemented for European and American options (EuropeanOption or AmericanOption instances)."""

    TOLERANCE = 1e-10 # class-level constant for floating-point comparisons

    def __init__(self, num_tree_steps: int):
        self.num_tree_steps = num_tree_steps

        # Input data validation
        self._validate_num_tree_steps()

    # Data validation method:
    def _validate_num_tree_steps(self):
        if not isinstance(self.num_tree_steps, int):
            raise TypeError(f"num_tree_steps must be an integer (type int), got {type(self.num_tree_steps)}")
        if self.num_tree_steps < 1:
            raise ValueError("num_tree_steps must be at least 1")    
    
    # The core of this class - pricing method:
    def _calculate_price(self, option: BaseOption, market_env: MarketEnvironment) -> float:
        T_minus_t = option.time_to_maturity(market_env.pricing_date)

        if T_minus_t < self.TOLERANCE: # Get option value at expiry
            return float(option.get_payoff(market_env.spot_price))
        elif market_env.volatility > self.TOLERANCE: # Normal case: some time remains to maturity and volatility is not zero
            # Implementing the CRR model (1979)
            dt = T_minus_t/self.num_tree_steps

            u = np.exp(market_env.volatility * np.sqrt(dt))
            d = 1/u
            p = (np.exp((market_env.risk_free_rate - market_env.dividend_yield) * dt) - d)/(u - d)

            if not (0.0 <= p <= 1.0):
                raise ValueError(f"Probability of an up-move must be between 0 and 1 and it is currently p={p:.4f}. Increase num_tree_steps.")

            j = np.arange(self.num_tree_steps + 1)

            # Terminal asset prices - ndarray
            S_terminal_array = market_env.spot_price * u**(2 * j - self.num_tree_steps) # only valid since d=1/u

            # Checking for overflow in terminal asset values
            if not np.all(np.isfinite(S_terminal_array)):
                raise ValueError(f"This combination of parameters caused a numerical overflow in terminal asset values. Please double-check if the volatility you specified (annualized vol={market_env.volatility*100:.2f}%) is correct.")

            # Terminal payoffs
            V_array = np.array(option.get_payoff(S_terminal_array))

            if isinstance(option, EuropeanOption):
                # Expected discounted payoff = option value for European options
                binom_probs = binom.pmf(j, self.num_tree_steps, p)
                option_value = np.sum(binom_probs * V_array) * np.exp(-market_env.risk_free_rate * T_minus_t)
            elif isinstance(option, AmericanOption):
                df = np.exp(-market_env.risk_free_rate * dt) # one-time-period discount factor
                # for American options, we must construct the tree and value down the tree
                for time in range((self.num_tree_steps-1), -1, -1):
                    j = np.arange(time + 1)
                    S_array_time = market_env.spot_price * u**(2 * j - time) # only valid since d=1/u
                    V_array = (V_array[1:] * p + V_array[:-1] * (1 - p)) * df
                    V_array = np.maximum(V_array, option.get_payoff(S_array_time)) # for American options, exercise if the current payoff is greater than the continuation value
                option_value = V_array[0] 
            else:
                raise TypeError("Binomial Tree pricing is implemented only for EuropeanOption or AmericanOption instances!")        

            return float(option_value)

        else: # Theoretical case option's value: when volatility is zero  
            if isinstance(option, EuropeanOption):
                df = np.exp(-market_env.risk_free_rate * T_minus_t)
                terminal_spot = market_env.spot_price * np.exp((market_env.risk_free_rate - market_env.dividend_yield) * T_minus_t)
                discounted_payoff = df * option.get_payoff(terminal_spot)

                return float(discounted_payoff)
            elif isinstance(option, AmericanOption):
                # Compute the present value of all discounted future payoffs (known with zero vol) and pick the highest payoff - exercise there
                dt = T_minus_t/self.num_tree_steps
                j = np.arange(self.num_tree_steps + 1)
                S_array = market_env.spot_price * np.exp((market_env.risk_free_rate - market_env.dividend_yield) * dt * j)
                dfs = np.exp(-market_env.risk_free_rate * dt * j)
                discounted_payoffs = dfs * option.get_payoff(S_array)

                return float(np.max(discounted_payoffs))
            else:
                raise TypeError("Binomial Tree pricing is implemented only for EuropeanOption or AmericanOption instances!") 
                          


