from dataclasses import dataclass
from datetime import date 

@dataclass
class MarketEnvironment:
    """Class storing the current market conditions, as of the date of pricing the option."""
    spot_price: float
    risk_free_rate: float
    volatility: float
    dividend_yield: float
    pricing_date: date


