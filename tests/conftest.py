import pytest
from datetime import date

from core.enums import OptionType
from instruments.european import EuropeanOption
from market.environment import MarketEnvironment

#----------------------------- MarketEnvironment instances
@pytest.fixture
def market_env_1():
    return MarketEnvironment(spot_price=100, risk_free_rate=0.05, volatility=0.25, dividend_yield=0.025, pricing_date=date(2026, 3, 18))

@pytest.fixture
def market_env_3():
    return MarketEnvironment(spot_price=100, risk_free_rate=0.02, volatility=0.4, dividend_yield=0.05, pricing_date=date(2026, 3, 18))

#----------------------------- EuropeanOption instances
@pytest.fixture
def european_call_option_1():
    return EuropeanOption(strike_price=100, expiry_date=date(2027, 3, 18), option_type=OptionType.CALL)

@pytest.fixture
def european_put_option_1():
    return EuropeanOption(strike_price=90, expiry_date=date(2027, 3, 18), option_type=OptionType.PUT)

@pytest.fixture
def european_call_option_3():
    return EuropeanOption(strike_price=90, expiry_date=date(2027, 4, 28), option_type=OptionType.CALL)

@pytest.fixture
def european_put_option_3():
    return EuropeanOption(strike_price=90, expiry_date=date(2027, 4, 28), option_type=OptionType.PUT)

@pytest.fixture
def european_call_option_4():
    return EuropeanOption(strike_price=100, expiry_date=date(2026, 9, 16), option_type=OptionType.CALL)


