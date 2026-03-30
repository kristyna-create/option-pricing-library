from __future__ import annotations
from typing import TYPE_CHECKING

from datetime import date
from abc import ABC, abstractmethod # for abstract classes and abstract methods
import math
import numpy as np

# other imports - for classes we will yet implement: BasePricer, MarketEnvironment, Greeks
if TYPE_CHECKING:
    from core.pricer_base import BasePricer
from core.greeks_data import Greeks
from market.environment import MarketEnvironment


# this will be an abstract class
# it will inherit from ABC class (from module abc)
# use @abstractmethod decorator on methods that subclasses must implement on their own - abstract methods
class BaseOption(ABC):
    def __init__(self, strike_price: float | int, expiry_date: date):
        self.strike_price = strike_price # later this might have to be implemented at the subclasses level (e.g. Asian option with floating strike)
        self.expiry_date = expiry_date

        # Input data validation
        self._validate_strike_price()
        self._validate_expiry_date()

    # data validation methods:
    def _validate_strike_price(self):
        if not isinstance(self.strike_price, (int, float)):
            raise TypeError(f"strike_price must be a number (float or integer), got {type(self.strike_price)}")
        if not math.isfinite(self.strike_price):
            raise ValueError("strike_price cannot be infinite or NaN")
        if self.strike_price <= 0.0: 
            raise ValueError("strike_price must be positive")
        
    def _validate_expiry_date(self):
        if not isinstance(self.expiry_date, date):
            raise TypeError(f"expiry_date must be a date (datetime.date), got {type(self.expiry_date)}") 

    # price() - concrete method, just delegates to the pricer to calculate the price of the option
    def price(self, pricer: BasePricer, market_env: MarketEnvironment) -> float:
        return pricer._calculate_price(option=self, market_env=market_env)

    # greeks() - concrete method, just delegates to the pricer to calculate Greeks
    def greeks(self, pricer: BasePricer, market_env: MarketEnvironment) -> Greeks:
        return pricer._calculate_greeks(option=self, market_env=market_env)

    # time_to_maturity() - concrete method, same for all options
    # for now, just implementing a simple day count convention
    def time_to_maturity(self, pricing_date: date) -> float:
        if not isinstance(pricing_date, date):
            raise TypeError(f"pricing_date must be a date (datetime.date), got {type(self.pricing_date)}") 
        if self.expiry_date < pricing_date:
            raise ValueError("pricing_date is after the option's expiry_date, please double-check your inputs")
        return (self.expiry_date - pricing_date).days / 365

    # get_payoff() - abstract method, different for each option
    @abstractmethod
    def get_payoff(self, spot_price: float | np.ndarray) -> float | np.ndarray:
        pass
