import numpy as np
from datetime import date
import math

from core.option_base import BaseOption
from core.enums import OptionType, AsianType

class AsianOption(BaseOption):
    """Class representing Asian options. strike_price must be specified for fixed strike Asian options (asian_type = AsianType.FIXED_STRIKE)
    and should not be specified for floating strike Asian options (asian_type = AsianType.FLOATING_STRIKE)."""
    def __init__(self, expiry_date: date, option_type: OptionType, asian_type: AsianType, strike_price: int | float | None = None):
        super().__init__(expiry_date = expiry_date)
        self.option_type = option_type
        self.asian_type = asian_type
        self.strike_price = strike_price

        # Input data validation
        self._validate_option_type()
        self._validate_asian_type()
        self._validate_strike_price()

    # Input data validation methods
    def _validate_option_type(self):
        if not isinstance(self.option_type, OptionType):
            valid_types = ", ".join(str(t) for t in OptionType)
            raise TypeError(f"option_type must be one of: {valid_types}, got {repr(self.option_type)}") # repr() gives the precise representation of an object, good for debugging
        
    def _validate_asian_type(self):
        if not isinstance(self.asian_type, AsianType):
            valid_types = ", ".join(str(t) for t in AsianType)
            raise TypeError(f"asian_type must be one of: {valid_types}, got {repr(self.asian_type)}")   
        
    def _validate_strike_price(self):
        if self.asian_type == AsianType.FIXED_STRIKE:
            if self.strike_price is None:
                raise ValueError(f"strike_price cannot be {self.strike_price}, you must specify it for {AsianType.FIXED_STRIKE.value} Asian options")
            if not isinstance(self.strike_price, (int, float)):
                raise TypeError(f"strike_price must be a number (float or integer), got {type(self.strike_price)}")
            if not math.isfinite(self.strike_price):
                raise ValueError("strike_price cannot be infinite or NaN")
            if self.strike_price <= 0.0: 
                raise ValueError("strike_price must be positive") 
        elif self.asian_type == AsianType.FLOATING_STRIKE:
            if self.strike_price is not None:
                raise ValueError(f"strike_price must be None for for {AsianType.FLOATING_STRIKE.value} Asian options and you inserted {self.strike_price}")      
            
    # Essential get_payoff() method - for Asian options
    def get_payoff(self, price_paths: np.ndarray) -> np.ndarray:
        """Computes the payoff of the Asian option based on the average price over the path.
            Fixed strike call: max(average_price - strike_price, 0)
            Fixed strike put: max(strike_price - average_price, 0)
            Floating strike call: max(terminal_price - average_price, 0)
            Floating strike put: max(average_price - terminal_price, 0)

            Args:
                price_paths: Simulated asset price path(s). 
                The expected shape is (num_paths, num_steps) where each row represents 
                one simulated path and each column represents one time step. The average is computed 
                along axis=1 (across time steps for each path).
                
            Returns:
                The option payoff(s) as a 1D array of shape (num_paths,), one
                payoff per simulated path.
            """
        # fixed strike Asian payoff
        if self.asian_type == AsianType.FIXED_STRIKE:
            if self.option_type == OptionType.CALL:
                return np.maximum(np.mean(price_paths, axis=1) - self.strike_price, 0.0)
            elif self.option_type == OptionType.PUT:
                return np.maximum(self.strike_price - np.mean(price_paths, axis=1), 0.0)
            else:
                valid_types = ", ".join(str(t) for t in OptionType)
                raise ValueError(f"get_payoff() method is currently implemented only for {valid_types} and you inserted {repr(self.option_type)}!")   
        # floating strike Asian payoff         
        elif self.asian_type == AsianType.FLOATING_STRIKE:
            if self.option_type == OptionType.CALL:
                return np.maximum(price_paths[:, -1] - np.mean(price_paths, axis=1), 0.0)  # price_paths[:, -1] - for each path, get the terminal price
            elif self.option_type == OptionType.PUT:
                return np.maximum(np.mean(price_paths, axis=1) - price_paths[:, -1], 0.0)  
            else:
                valid_types = ", ".join(str(t) for t in OptionType)
                raise ValueError(f"get_payoff() method is currently implemented only for {valid_types} and you inserted {repr(self.option_type)}!") 
        else:
            valid_types = ", ".join(str(t) for t in AsianType)
            raise ValueError(f"get_payoff() method is currently implemented only for {valid_types} and you inserted {repr(self.asian_type)}!")      