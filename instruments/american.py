import numpy as np
from datetime import date
import math

from core.option_base import BaseOption
from core.enums import OptionType

class AmericanOption(BaseOption):
    def __init__(self, expiry_date: date, option_type: OptionType, strike_price: int | float):
        super().__init__(expiry_date = expiry_date)
        self.option_type = option_type
        self.strike_price = strike_price

        # Input data validation
        self._validate_option_type()
        self._validate_strike_price()

    # Input data validation methods
    def _validate_option_type(self):
        if not isinstance(self.option_type, OptionType):
            valid_types = ", ".join(str(t) for t in OptionType)
            raise TypeError(f"option_type must be one of: {valid_types}, got {repr(self.option_type)}") # repr() gives the precise representation of an object, good for debugging
        
    def _validate_strike_price(self):
        if not isinstance(self.strike_price, (int, float)):
            raise TypeError(f"strike_price must be a number (float or integer), got {type(self.strike_price)}")
        if not math.isfinite(self.strike_price):
            raise ValueError("strike_price cannot be infinite or NaN")
        if self.strike_price <= 0.0: 
            raise ValueError("strike_price must be positive")    
        
    # Essential get_payoff() method - for American options    
    def get_payoff(self, spot_price: float | np.ndarray) -> float | np.ndarray:
        """Computes the payoff of the American option.
            For a call: max(spot_price - strike_price, 0)
            For a put: max(strike_price - spot_price, 0)

            Args:
                spot_price: The underlying asset price at the current point in time. Accepts a single
                    float for scalar pricing or a numpy array for vectorized pricing
                    across multiple spot prices (e.g. for a binomial tree).
                    
            Returns:
                The option payoff(s), matching the type and shape of spot_price.
            """
        if self.option_type == OptionType.CALL:
            return np.maximum(spot_price - self.strike_price, 0.0)
        elif self.option_type == OptionType.PUT:
            return np.maximum(self.strike_price - spot_price, 0.0)
        else:
            valid_types = ", ".join(str(t) for t in OptionType)
            raise ValueError(f"get_payoff() method is currently implemented only for {valid_types} and you inserted {repr(self.option_type)}!")    