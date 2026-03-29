import numpy as np
from scipy.stats import binom
from dataclasses import replace
from datetime import timedelta

from core.pricer_base import BasePricer
from core.option_base import BaseOption
from core.enums import OptionType
from instruments.european import EuropeanOption
from market.environment import MarketEnvironment
from core.greeks_data import Greeks

class BinomialTreePricer(BasePricer):

    tolerance = 1e-10 # for floating-point comparisons

    def __init__(self, num_tree_steps: int):
        self.num_tree_steps = num_tree_steps

        # Input data validation
        self._validate_num_tree_steps()

    # data validation method:
    def _validate_num_tree_steps(self):
        if not isinstance(self.num_tree_steps, int):
            raise TypeError(f"num_tree_steps must be an integer (type int), got {type(self.num_tree_steps)}")
        if self.num_tree_steps < 1:
            raise ValueError("num_tree_steps must be at least 1")    

    def _calculate_price(self, option: BaseOption, market_env: MarketEnvironment) -> float:
        if isinstance(option, EuropeanOption):
            T_minus_t = option.time_to_maturity(market_env.pricing_date)

            if T_minus_t < self.tolerance: # Get option value at expiry
                return option.get_payoff(market_env.spot_price)
            elif market_env.volatility > self.tolerance: # Normal case: some time remains to maturity and volatility is not zero
                # Implementing the CRR model (1979)
                dt = T_minus_t/self.num_tree_steps

                u = np.exp(market_env.volatility * np.sqrt(dt))
                d = 1/u
                p = (np.exp((market_env.risk_free_rate - market_env.dividend_yield) * dt) - d)/(u - d)

                if not (0.0 <= p <= 1.0):
                    raise ValueError(f"Probability of an up-move must be between 0 and 1 and it is currently p={p:.4f}. Increase num_tree_steps.")

                j = np.arange(self.num_tree_steps + 1)

                # Terminal asset prices
                S_terminal_array = market_env.spot_price * u ** (2 * j - self.num_tree_steps) # only valid since d=1/u

                # Terminal payoffs
                if option.option_type == OptionType.CALL:
                    V_array = np.maximum(S_terminal_array - option.strike_price, 0)
                elif option.option_type == OptionType.PUT:
                    V_array = np.maximum(option.strike_price - S_terminal_array, 0)
                else:
                    valid_types = ", ".join(str(t) for t in OptionType)
                    raise ValueError(f"Pricing EuropeanOption via Binomial Tree is currently implemented only for {valid_types} and you inserted {repr(option.option_type)}!")

                # Expected discounted payoff = option value for European options
                binom_probs = binom.pmf(j, self.num_tree_steps, p)
                option_value = np.sum(binom_probs * V_array) * np.exp(-market_env.risk_free_rate * T_minus_t)

                return float(option_value)

            else: # Theoretical case option's value: when volatility is zero
                return float(np.exp(-market_env.risk_free_rate * T_minus_t) * option.get_payoff(market_env.spot_price * np.exp((market_env.risk_free_rate - market_env.dividend_yield) * T_minus_t)))     

        else:
            raise TypeError("Binomial Tree pricing is currently implemented only for EuropeanOption instances!")

    def _calculate_greeks(self, option: BaseOption, market_env: MarketEnvironment) -> Greeks:
        if not isinstance(option, EuropeanOption):
            raise TypeError("Binomial Tree pricing is currently implemented only for EuropeanOption instances!")
        
        # base case price using current market environment
        price = self._calculate_price(option, market_env)

        # delta
        # using central difference
        delta_S = 0.01 * market_env.spot_price
        market_env_S_plus = replace(market_env, spot_price = market_env.spot_price + delta_S)
        market_env_S_minus = replace(market_env, spot_price = market_env.spot_price - delta_S)
        price_plus = self._calculate_price(option, market_env_S_plus)
        price_minus = self._calculate_price(option, market_env_S_minus)

        delta = (price_plus - price_minus)/(2*delta_S)

        # gamma
        # computing central difference
        gamma = (price_plus - 2 * price + price_minus)/delta_S**2

        # vega
        delta_vol = 0.001
        market_env_vol_plus = replace(market_env, volatility = market_env.volatility + delta_vol)
        price_plus = self._calculate_price(option, market_env_vol_plus)
        # using central difference if (vol - delta_vol) is not negative
        if (market_env.volatility - delta_vol) >= 0.0:
            market_env_vol_minus = replace(market_env, volatility = market_env.volatility - delta_vol)
            price_minus = self._calculate_price(option, market_env_vol_minus)
            vega = (price_plus - price_minus)/(2*delta_vol)
        else: # using forward difference
            vega = (price_plus - price)/delta_vol

        # theta
        # shift the pricing date forward by one day
        dt = 1/365 # annualized theta
        
        if (market_env.pricing_date + timedelta(days=1)) > option.expiry_date: # pricing_date cannot be after expiry_date
            price_tomorrow = option.get_payoff(market_env.spot_price) # get option value at expiry
            # numerical theta at expiry will be zero by construction 
        else:    
            market_env_tomorrow = replace(market_env, pricing_date = market_env.pricing_date + timedelta(days=1))
            price_tomorrow = self._calculate_price(option, market_env_tomorrow)

        theta = (price_tomorrow - price)/dt

        # rho
        # using central difference
        delta_r = 0.001
        market_env_r_plus = replace(market_env, risk_free_rate = market_env.risk_free_rate + delta_r)
        market_env_r_minus = replace(market_env, risk_free_rate = market_env.risk_free_rate - delta_r) # we can have negative interest rates
        price_plus = self._calculate_price(option, market_env_r_plus)
        price_minus = self._calculate_price(option, market_env_r_minus)
        
        rho = (price_plus - price_minus)/(2*delta_r)

        return Greeks(delta, gamma, vega, theta, rho)
