import pytest 
from datetime import date
import math

from core.enums import OptionType
from instruments.european import EuropeanOption
from market.environment import MarketEnvironment

from engines.black_scholes_merton import BlackScholesMertonPricer

#----------------------------- Define fixtures
# pytest fixtures @pytest.fixture: extract the common market environments and european options into fixtures
# fixtures also in conftest.py
#--------------- MarketEnvironment instances
@pytest.fixture
def market_env_2():
    return MarketEnvironment(spot_price=100, risk_free_rate=0.02, volatility=0.0, dividend_yield=0.05, pricing_date=date(2026, 3, 18))

#--------------- EuropeanOption instances
@pytest.fixture
def european_call_option_2():
    return EuropeanOption(strike_price=100, expiry_date=date(2028, 9, 16), option_type=OptionType.CALL)

@pytest.fixture
def european_put_option_2():
    return EuropeanOption(strike_price=100, expiry_date=date(2028, 9, 16), option_type=OptionType.PUT)


#----------------------------- Option values test - known values
# known values are obtained from the online BSM calculator here: https://www.omnicalculator.com/finance/black-scholes and here: https://quantpie.co.uk/oup/oup_bsm_price_greeks.php

def test_call_pricing_known_value(market_env_1, european_call_option_1):
    assert european_call_option_1.price(pricer=BlackScholesMertonPricer(), market_env=market_env_1) == pytest.approx(10.83, abs=0.01)

def test_put_pricing_known_value(market_env_1, european_put_option_1):
    assert european_put_option_1.price(pricer=BlackScholesMertonPricer(), market_env=market_env_1) == pytest.approx(4.35, abs=0.01)    

def test_call_pricing_expiry():
    test_market = MarketEnvironment(spot_price=120, risk_free_rate=0.05, volatility=0.25, dividend_yield=0.025, pricing_date=date(2026, 3, 18)) # unique, fixture not defined
    test_call_option = EuropeanOption(strike_price=100, expiry_date=date(2026, 3, 18), option_type=OptionType.CALL) # unique, fixture not defined

    assert test_call_option.price(pricer=BlackScholesMertonPricer(), market_env=test_market) == pytest.approx(20.0, abs=0.01)

def test_put_pricing_expiry():
    test_market = MarketEnvironment(spot_price=95, risk_free_rate=0.05, volatility=0.25, dividend_yield=0.025, pricing_date=date(2026, 3, 18)) # unique, fixture not defined
    test_put_option = EuropeanOption(strike_price=100, expiry_date=date(2026, 3, 18), option_type=OptionType.PUT) # unique, fixture not defined

    assert test_put_option.price(pricer=BlackScholesMertonPricer(), market_env=test_market) == pytest.approx(5.0, abs=0.01) 

def test_call_pricing_zero_vol(european_call_option_2):
    test_market = MarketEnvironment(spot_price=110, risk_free_rate=0.02, volatility=0.0, dividend_yield=0.05, pricing_date=date(2026, 3, 18)) # unique, fixture not defined
    test_price = european_call_option_2.price(pricer=BlackScholesMertonPricer(), market_env=test_market) # Compute the call value for the test, call will expire ITM

    assert test_price == pytest.approx(1.9477, abs=0.0001)    

def test_put_pricing_zero_vol(market_env_2, european_put_option_2):
    test_price = european_put_option_2.price(pricer=BlackScholesMertonPricer(), market_env=market_env_2) # Compute the put value for the test, put will expire ITM

    assert test_price == pytest.approx(6.8767, abs=0.0001)

#----------------------------- Put-Call parity test - remember dividend yield
def test_put_call_parity(market_env_1, european_put_option_1):
    test_call_option = EuropeanOption(strike_price=90, expiry_date=date(2027, 3, 18), option_type=OptionType.CALL) # this call option is unique, fixture not defined

    price_call = test_call_option.price(pricer=BlackScholesMertonPricer(), market_env=market_env_1)
    price_put = european_put_option_1.price(pricer=BlackScholesMertonPricer(), market_env=market_env_1)
    T_minus_t = test_call_option.time_to_maturity(market_env_1.pricing_date)

    assert (price_call - price_put) == pytest.approx(market_env_1.spot_price * math.exp(-market_env_1.dividend_yield * T_minus_t) - test_call_option.strike_price * math.exp(-market_env_1.risk_free_rate * T_minus_t), abs=0.0001)


#----------------------------- Test input data validation: e.g. creating a MarketEnvironment with a negative spot price - pytest.raises
@pytest.mark.parametrize("market_env_inputs, error_type, exception_str", [
    ({"spot_price": -100, "risk_free_rate": 0.05, "volatility": 0.25, "dividend_yield": 0.025, "pricing_date": date(2026, 3, 18)}, ValueError, "spot_price must be positive"),
    ({"spot_price": 100, "risk_free_rate": "0.05", "volatility": 0.25, "dividend_yield": 0.025, "pricing_date": date(2026, 3, 18)}, TypeError, "risk_free_rate must be a number"),
    ({"spot_price": 100, "risk_free_rate": 0.05, "volatility": float("inf"), "dividend_yield": 0.025, "pricing_date": date(2026, 3, 18)}, ValueError, "volatility cannot be infinite")
    ])
def test_inputs(market_env_inputs, error_type, exception_str):
    with pytest.raises(error_type) as exc_info:
        MarketEnvironment(**market_env_inputs)
    assert exception_str in str(exc_info.value)        

#----------------------------- Greek values test
# known values are obtained from the online BSM calculator here: https://quantpie.co.uk/oup/oup_bsm_price_greeks.php   

def test_call_Greeks_known_value(european_call_option_2, market_env_3):
    test_greeks = european_call_option_2.greeks(pricer=BlackScholesMertonPricer(), market_env=market_env_3) # Compute the Greeks for the test

    assert test_greeks.delta == pytest.approx(0.5104, abs=0.0001)
    assert test_greeks.gamma == pytest.approx(0.0055, abs=0.0001)
    assert test_greeks.vega == pytest.approx(54.6004, abs=0.0001)
    assert test_greeks.theta == pytest.approx(-2.4450, abs=0.0001)
    assert test_greeks.rho == pytest.approx(78.9469, abs=0.0001)

def test_put_Greeks_known_value(european_put_option_2, market_env_3):
    test_greeks = european_put_option_2.greeks(pricer=BlackScholesMertonPricer(), market_env=market_env_3) # Compute the Greeks for the test

    assert test_greeks.delta == pytest.approx(-0.3721, abs=0.0001)
    assert test_greeks.gamma == pytest.approx(0.0055, abs=0.0001)
    assert test_greeks.vega == pytest.approx(54.6004, abs=0.0001)
    assert test_greeks.theta == pytest.approx(-4.9548, abs=0.0001)
    assert test_greeks.rho == pytest.approx(-158.9842, abs=0.0001)

def test_call_Greeks_zero_vol(european_call_option_2, market_env_2):
    test_greeks = european_call_option_2.greeks(pricer=BlackScholesMertonPricer(), market_env=market_env_2) # Compute the Greeks for the test, call will expire OTM

    assert test_greeks.delta == pytest.approx(0, abs=0.0001)
    assert test_greeks.gamma == pytest.approx(0, abs=0.0001)
    assert test_greeks.vega == pytest.approx(0, abs=0.0001)
    assert test_greeks.theta == pytest.approx(0, abs=0.0001)
    assert test_greeks.rho == pytest.approx(0, abs=0.0001)

def test_put_Greeks_zero_vol(market_env_2, european_put_option_2):
    test_greeks = european_put_option_2.greeks(pricer=BlackScholesMertonPricer(), market_env=market_env_2) # Compute the Greeks for the test, put will expire ITM

    assert test_greeks.delta == pytest.approx(-0.8824, abs=0.0001)
    assert test_greeks.gamma == pytest.approx(0, abs=0.0001)
    assert test_greeks.vega == pytest.approx(0, abs=0.0001)
    assert test_greeks.theta == pytest.approx(-2.5098, abs=0.0001)
    assert test_greeks.rho == pytest.approx(-237.9311, abs=0.0001)    