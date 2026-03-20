from datetime import date

from core.option_base import BaseOption
from core.enums import OptionType

class EuropeanOption(BaseOption): # EuropeanOption is a subclass of BaseOption
    def __init__(self, strike_price: float, expiry_date: date, option_type: OptionType):
        super().__init__(strike_price=strike_price, expiry_date=expiry_date)
        self.option_type = option_type
        
        # Input data validation
        self._validate_option_type()

    # Input data validation method
    def _validate_option_type(self):
        if not isinstance(self.option_type, OptionType):
            valid_types = ", ".join(str(t) for t in OptionType)
            raise TypeError(f"option_type must be one of: {valid_types}, got {repr(self.option_type)}") # repr() gives the precise representation of an object, good for debugging
    
    # Essential get_payoff() method - here for European options
    def get_payoff(self, spot_price: float) -> float:
        if self.option_type == OptionType.CALL:
            return max(spot_price - self.strike_price, 0.0)
        elif self.option_type == OptionType.PUT:
            return max(self.strike_price - spot_price, 0.0)
        else:
            valid_types = ", ".join(str(t) for t in OptionType)
            raise ValueError(f"get_payoff() method is currently implemented only for {valid_types} and you inserted {repr(self.option_type)}!")