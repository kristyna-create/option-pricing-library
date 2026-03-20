from dataclasses import dataclass
from datetime import date 
import math
import warnings

@dataclass
class MarketEnvironment:
    """Class storing the current market conditions, as of the date of pricing the option."""
    spot_price: float | int
    risk_free_rate: float | int
    volatility: float | int
    dividend_yield: float | int
    pricing_date: date

    # Validation of input data using __post_init__
    def __post_init__(self):
        self._validate_spot_price()
        self._validate_risk_free_rate()
        self._validate_volatility()
        self._validate_dividend_yield()
        self._validate_pricing_date()

    def _validate_spot_price(self):
        """Spot price must be positive."""
        if not isinstance(self.spot_price, (int, float)):
            raise TypeError(f"spot_price must be a number (float or integer), got {type(self.spot_price)}")
        if not math.isfinite(self.spot_price):
            raise ValueError("spot_price cannot be infinite or NaN")
        if self.spot_price <= 0.0: 
            raise ValueError("spot_price must be positive")

    def _validate_risk_free_rate(self):
        if not isinstance(self.risk_free_rate, (int, float)):
            raise TypeError(f"risk_free_rate must be a number (float or integer), got {type(self.risk_free_rate)}")
        if not math.isfinite(self.risk_free_rate):
            raise ValueError("risk_free_rate cannot be infinite or NaN")
        if abs(self.risk_free_rate) >= 1.0:
            warnings.warn(f"risk_free_rate is expected to be in decimal form, so the computation will be done with the risk_free_rate = {self.risk_free_rate*100}%") 

    def _validate_volatility(self):
        if not isinstance(self.volatility, (int, float)):
            raise TypeError(f"volatility must be a number (float or integer), got {type(self.volatility)}")
        if not math.isfinite(self.volatility):
            raise ValueError("volatility cannot be infinite or NaN")
        if self.volatility < 0.0: 
            raise ValueError("volatility cannot be negative")   
        if self.volatility >= 1.0:
            warnings.warn(f"volatility is expected to be in decimal form, so the computation will be done with the volatility = {self.volatility*100}%")   

    def _validate_dividend_yield(self):
        if not isinstance(self.dividend_yield, (int, float)): 
            raise TypeError(f"dividend_yield must be a number (float or integer), got {type(self.dividend_yield)}")
        if not math.isfinite(self.dividend_yield):
            raise ValueError("dividend_yield cannot be infinite or NaN")
        if self.dividend_yield < 0.0:
            raise ValueError("dividend_yield cannot be negative")    
        if self.dividend_yield >= 1.0:
            warnings.warn(f"dividend_yield is expected to be in decimal form, so the computation will be done with the dividend_yield = {self.dividend_yield*100}%")   

    def _validate_pricing_date(self):
        if not isinstance(self.pricing_date, date):
            raise TypeError(f"pricing_date must be a date (datetime.date), got {type(self.pricing_date)}")            



