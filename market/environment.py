from dataclasses import dataclass
from datetime import date 
import numpy as np
import warnings

@dataclass
class MarketEnvironment:
    """Class storing the current market conditions, as of the date of pricing the option.
    Allows for vectorized pricing via allowing for numpy ndarray inputs (except for the pricing_date).
    """
    spot_price: int | float | np.ndarray
    risk_free_rate: int | float | np.ndarray
    volatility: int | float | np.ndarray
    pricing_date: date
    dividend_yield: int | float | np.ndarray = 0.0

    # Validation of input data using __post_init__ (for dataclasses)
    def __post_init__(self):
        self._validate_spot_price()
        self._validate_risk_free_rate()
        self._validate_volatility()
        self._validate_dividend_yield()
        self._validate_pricing_date()
        self._validate_shapes() # added for vectorized pricing

    def _validate_spot_price(self):
        if not isinstance(self.spot_price, (int, float, np.ndarray)):
            raise TypeError(f"spot_price must be a number (float or integer) or a numpy array (numpy.ndarray) of numbers, got {type(self.spot_price)}")
        if isinstance(self.spot_price, np.ndarray):
            is_real_numeric = (np.issubdtype(self.spot_price.dtype, np.integer) or np.issubdtype(self.spot_price.dtype, np.floating)) # allow for integers or floats only
            if not is_real_numeric:
                raise TypeError(f"Array of spot prices must contain only real numbers (int/float), got: {self.spot_price.dtype}")
        if not np.isfinite(self.spot_price).all():
            if isinstance(self.spot_price, np.ndarray):
                raise ValueError("The spot_price array contains one or more non-finite values (inf or NaN).")
            else:
                raise ValueError(f"spot_price cannot be infinite or NaN, got: {self.spot_price}")
        if np.any(self.spot_price <= 0.0):
            if isinstance(self.spot_price, np.ndarray):
                raise ValueError("One or more values in the spot_price array are <= 0. All elements must be strictly positive.")
            else:
                raise ValueError(f"spot_price must be positive, got: {self.spot_price}")    

    def _validate_risk_free_rate(self):
        if not isinstance(self.risk_free_rate, (int, float, np.ndarray)):
            raise TypeError(f"risk_free_rate must be a number (float or integer) or a numpy array (numpy.ndarray) of numbers, got {type(self.risk_free_rate)}")
        if isinstance(self.risk_free_rate, np.ndarray):
            is_real_numeric = (np.issubdtype(self.risk_free_rate.dtype, np.integer) or np.issubdtype(self.risk_free_rate.dtype, np.floating)) # allow for integers or floats only
            if not is_real_numeric:
                raise TypeError(f"Array of risk free rates must contain only real numbers (int/float), got: {self.risk_free_rate.dtype}")
        if not np.isfinite(self.risk_free_rate).all():
            if isinstance(self.risk_free_rate, np.ndarray):
                raise ValueError("The risk_free_rate array contains one or more non-finite values (inf or NaN).")
            else:
                raise ValueError(f"risk_free_rate cannot be infinite or NaN, got: {self.risk_free_rate}")
        if np.any(np.abs(self.risk_free_rate) >= 1.0):
            if isinstance(self.risk_free_rate, np.ndarray):
                warnings.warn("One or more values in risk_free_rate array are >= 1.0 (in the absolute value). Ensure rates are in decimal form (e.g., 0.05 for 5%).")
            else:
                warnings.warn(
                    f"risk_free_rate is expected to be in decimal form."
                    f"Computation will proceed with {self.risk_free_rate * 100}%."
                )

    def _validate_volatility(self):
        if not isinstance(self.volatility, (int, float, np.ndarray)):
            raise TypeError(f"volatility must be a number (float or integer) or a numpy array (numpy.ndarray) of numbers, got {type(self.volatility)}")
        if isinstance(self.volatility, np.ndarray):
            is_real_numeric = (np.issubdtype(self.volatility.dtype, np.integer) or np.issubdtype(self.volatility.dtype, np.floating)) # allow for integers or floats only
            if not is_real_numeric:
                raise TypeError(f"Array of volatilities must contain only real numbers (int/float), got: {self.volatility.dtype}")
        if not np.isfinite(self.volatility).all():
            if isinstance(self.volatility, np.ndarray):
                raise ValueError("The volatility array contains one or more non-finite values (inf or NaN).")
            else:
                raise ValueError(f"volatility cannot be infinite or NaN, got: {self.volatility}")
        if np.any(self.volatility < 0.0):
            if isinstance(self.volatility, np.ndarray):
                raise ValueError("One or more values in the volatility array are negative. All elements must be positive or zero.")
            else:
                raise ValueError(f"volatility cannot be negative, got: {self.volatility}")    
        if np.any(self.volatility >= 1.0):
            if isinstance(self.volatility, np.ndarray):
                warnings.warn("One or more values in volatility array are >= 1.0. Ensure volatilities are in decimal form (e.g., 0.05 for 5%).")
            else:
                warnings.warn(
                    f"volatility is expected to be in decimal form."
                    f"Computation will proceed with {self.volatility * 100}%."
                )  

    def _validate_dividend_yield(self):
        if not isinstance(self.dividend_yield, (int, float, np.ndarray)): 
            raise TypeError(f"dividend_yield must be a number (float or integer) or a numpy array (numpy.ndarray) of numbers, got {type(self.dividend_yield)}")
        if isinstance(self.dividend_yield, np.ndarray):
            is_real_numeric = (np.issubdtype(self.dividend_yield.dtype, np.integer) or np.issubdtype(self.dividend_yield.dtype, np.floating)) # allow for integers or floats only
            if not is_real_numeric:
                raise TypeError(f"Array of dividend yields must contain only real numbers (int/float), got: {self.dividend_yield.dtype}")
        if not np.isfinite(self.dividend_yield).all():
            if isinstance(self.dividend_yield, np.ndarray):
                raise ValueError("The dividend_yield array contains one or more non-finite values (inf or NaN).")
            else:
                raise ValueError(f"dividend_yield cannot be infinite or NaN, got: {self.dividend_yield}")
        if np.any(self.dividend_yield < 0.0):
            if isinstance(self.dividend_yield, np.ndarray):
                raise ValueError("One or more values in the dividend_yield array are negative. All elements must be positive or zero.")
            else:
                raise ValueError(f"dividend_yield cannot be negative, got: {self.dividend_yield}")     
        if np.any(np.abs(self.dividend_yield) >= 1.0):
            if isinstance(self.dividend_yield, np.ndarray):
                warnings.warn("One or more values in dividend_yield array are >= 1.0. Ensure dividend yields are in decimal form (e.g., 0.05 for 5%).")
            else:
                warnings.warn(
                    f"dividend_yield is expected to be in decimal form."
                    f"Computation will proceed with {self.dividend_yield * 100}%."
                )    

    def _validate_pricing_date(self):
        if not isinstance(self.pricing_date, date):
            raise TypeError(f"pricing_date must be a date (datetime.date), got {type(self.pricing_date)}") 

    def _validate_shapes(self):
        """Ensure all array-type inputs have the same shape for vectorized pricing."""
        arrays = {
            "spot_price": self.spot_price,
            "risk_free_rate": self.risk_free_rate,
            "volatility": self.volatility,
            "dividend_yield": self.dividend_yield
        }
        
        shapes = {}
        for name, value in arrays.items():
            # Checking only for arrays that have more than 1 element.
            # size 0 (empty) or size 1 (scalar-like) arrays are ignored.
            if isinstance(value, np.ndarray) and value.size > 1:
                shapes[name] = value.shape

        if len(shapes) > 1:
            it = iter(shapes.values())
            first_shape = next(it)
            if not all(s == first_shape for s in it):
                shape_summary = ", ".join([f"{k}: {v}" for k, v in shapes.items()])
                raise ValueError(
                    f"Inconsistent array shapes in MarketEnvironment. "
                    f"All arrays must have the same shape. Found {shape_summary}"
                )               



