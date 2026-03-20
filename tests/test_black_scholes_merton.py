import pytest 
from datetime import date
import math

from core.enums import OptionType
from instruments.european import EuropeanOption
from market.environment import MarketEnvironment

from engines.black_scholes_merton import BlackScholesMertonPricer

#----------------------------- Option values test
# known values are obtained from the online BSM calculator here: https://www.omnicalculator.com/finance/black-scholes and here: https://quantpie.co.uk/oup/oup_bsm_price_greeks.php

def test_call_pricing_known_value():
    test_market = MarketEnvironment(spot_price=100, risk_free_rate=0.05, volatility=0.25, dividend_yield=0.025, pricing_date=date(2026, 3, 18))
    test_call_option = EuropeanOption(strike_price=100, expiry_date=date(2027, 3, 18), option_type=OptionType.CALL)

    assert test_call_option.price(pricer=BlackScholesMertonPricer(), market_env=test_market) == pytest.approx(10.83, abs=0.01)

def test_put_pricing_known_value():
    test_market = MarketEnvironment(spot_price=100, risk_free_rate=0.05, volatility=0.25, dividend_yield=0.025, pricing_date=date(2026, 3, 18))
    test_put_option = EuropeanOption(strike_price=90, expiry_date=date(2027, 3, 18), option_type=OptionType.PUT)

    assert test_put_option.price(pricer=BlackScholesMertonPricer(), market_env=test_market) == pytest.approx(4.35, abs=0.01)    

def test_call_pricing_expiry():
    test_market = MarketEnvironment(spot_price=120, risk_free_rate=0.05, volatility=0.25, dividend_yield=0.025, pricing_date=date(2026, 3, 18))
    test_call_option = EuropeanOption(strike_price=100, expiry_date=date(2026, 3, 18), option_type=OptionType.CALL)

    assert test_call_option.price(pricer=BlackScholesMertonPricer(), market_env=test_market) == pytest.approx(20.0, abs=0.01)

def test_put_pricing_expiry():
    test_market = MarketEnvironment(spot_price=95, risk_free_rate=0.05, volatility=0.25, dividend_yield=0.025, pricing_date=date(2026, 3, 18))
    test_put_option = EuropeanOption(strike_price=100, expiry_date=date(2026, 3, 18), option_type=OptionType.PUT)

    assert test_put_option.price(pricer=BlackScholesMertonPricer(), market_env=test_market) == pytest.approx(5.0, abs=0.01) 

def test_call_pricing_zero_vol():
    test_market = MarketEnvironment(spot_price=110, risk_free_rate=0.02, volatility=0.0, dividend_yield=0.05, pricing_date=date(2026, 3, 18))
    test_call_option = EuropeanOption(strike_price=100, expiry_date=date(2028, 9, 16), option_type=OptionType.CALL)
    test_price = test_call_option.price(pricer=BlackScholesMertonPricer(), market_env=test_market) # Compute the call value for the test, call will expire ITM

    assert test_price == pytest.approx(1.9477, abs=0.0001)    

def test_put_pricing_zero_vol():
    test_market = MarketEnvironment(spot_price=100, risk_free_rate=0.02, volatility=0.0, dividend_yield=0.05, pricing_date=date(2026, 3, 18))
    test_put_option = EuropeanOption(strike_price=100, expiry_date=date(2028, 9, 16), option_type=OptionType.PUT)
    test_price = test_put_option.price(pricer=BlackScholesMertonPricer(), market_env=test_market) # Compute the put value for the test, put will expire ITM

    assert test_price == pytest.approx(6.8767, abs=0.0001)

#----------------------------- Put-Call parity test - remember dividend yield
def test_put_call_parity():
    test_market = MarketEnvironment(spot_price=100, risk_free_rate=0.05, volatility=0.25, dividend_yield=0.025, pricing_date=date(2026, 3, 18))
    test_call_option = EuropeanOption(strike_price=90, expiry_date=date(2027, 3, 18), option_type=OptionType.CALL)
    test_put_option = EuropeanOption(strike_price=90, expiry_date=date(2027, 3, 18), option_type=OptionType.PUT)

    price_call = test_call_option.price(pricer=BlackScholesMertonPricer(), market_env=test_market)
    price_put = test_put_option.price(pricer=BlackScholesMertonPricer(), market_env=test_market)
    T_minus_t = test_call_option.time_to_maturity(test_market.pricing_date)

    assert (price_call - price_put) == pytest.approx(test_market.spot_price * math.exp(-test_market.dividend_yield * T_minus_t) - test_call_option.strike_price * math.exp(-test_market.risk_free_rate * T_minus_t), abs=0.0001)


#----------------------------- Test input data validation: e.g. creating a MarketEnvironment with a negative spot price - pytest.raises
def test_negative_spot_validation():
    with pytest.raises(ValueError) as exc_info:
        MarketEnvironment(spot_price=-100, risk_free_rate=0.05, volatility=0.25, dividend_yield=0.025, pricing_date=date(2026, 3, 18))
    assert "spot_price" in str(exc_info.value)  

def test_risk_free_rate_validation():
    with pytest.raises(TypeError) as exc_info:
        MarketEnvironment(spot_price=100, risk_free_rate="0.05", volatility=0.25, dividend_yield=0.025, pricing_date=date(2026, 3, 18)) # type: ignore
    assert "risk_free_rate must be a number" in str(exc_info.value)    

def test_volatility_validation():
    with pytest.raises(ValueError) as exc_info:
        MarketEnvironment(spot_price=100, risk_free_rate=0.05, volatility=float("inf"), dividend_yield=0.025, pricing_date=date(2026, 3, 18))
    assert "volatility cannot be infinite" in str(exc_info)

#----------------------------- Greek values test
# known values are obtained from the online BSM calculator here: https://quantpie.co.uk/oup/oup_bsm_price_greeks.php   

def test_call_Greeks_known_value():
    test_market = MarketEnvironment(spot_price=100, risk_free_rate=0.02, volatility=0.4, dividend_yield=0.05, pricing_date=date(2026, 3, 18))
    test_call_option = EuropeanOption(strike_price=100, expiry_date=date(2028, 9, 16), option_type=OptionType.CALL)
    test_greeks = test_call_option.greeks(pricer=BlackScholesMertonPricer(), market_env=test_market) # Compute the Greeks for the test

    assert test_greeks.delta == pytest.approx(0.5104, abs=0.0001)
    assert test_greeks.gamma == pytest.approx(0.0055, abs=0.0001)
    assert test_greeks.vega == pytest.approx(54.6004, abs=0.0001)
    assert test_greeks.theta == pytest.approx(-2.4450, abs=0.0001)
    assert test_greeks.rho == pytest.approx(78.9469, abs=0.0001)

def test_put_Greeks_known_value():
    test_market = MarketEnvironment(spot_price=100, risk_free_rate=0.02, volatility=0.4, dividend_yield=0.05, pricing_date=date(2026, 3, 18))
    test_put_option = EuropeanOption(strike_price=100, expiry_date=date(2028, 9, 16), option_type=OptionType.PUT)
    test_greeks = test_put_option.greeks(pricer=BlackScholesMertonPricer(), market_env=test_market) # Compute the Greeks for the test

    assert test_greeks.delta == pytest.approx(-0.3721, abs=0.0001)
    assert test_greeks.gamma == pytest.approx(0.0055, abs=0.0001)
    assert test_greeks.vega == pytest.approx(54.6004, abs=0.0001)
    assert test_greeks.theta == pytest.approx(-4.9548, abs=0.0001)
    assert test_greeks.rho == pytest.approx(-158.9842, abs=0.0001)

def test_call_Greeks_zero_vol():
    test_market = MarketEnvironment(spot_price=100, risk_free_rate=0.02, volatility=0.0, dividend_yield=0.05, pricing_date=date(2026, 3, 18))
    test_call_option = EuropeanOption(strike_price=100, expiry_date=date(2028, 9, 16), option_type=OptionType.CALL)
    test_greeks = test_call_option.greeks(pricer=BlackScholesMertonPricer(), market_env=test_market) # Compute the Greeks for the test, call will expire OTM

    assert test_greeks.delta == pytest.approx(0, abs=0.0001)
    assert test_greeks.gamma == pytest.approx(0, abs=0.0001)
    assert test_greeks.vega == pytest.approx(0, abs=0.0001)
    assert test_greeks.theta == pytest.approx(0, abs=0.0001)
    assert test_greeks.rho == pytest.approx(0, abs=0.0001)

def test_put_Greeks_zero_vol():
    test_market = MarketEnvironment(spot_price=100, risk_free_rate=0.02, volatility=0.0, dividend_yield=0.05, pricing_date=date(2026, 3, 18))
    test_put_option = EuropeanOption(strike_price=100, expiry_date=date(2028, 9, 16), option_type=OptionType.PUT)
    test_greeks = test_put_option.greeks(pricer=BlackScholesMertonPricer(), market_env=test_market) # Compute the Greeks for the test, put will expire ITM

    assert test_greeks.delta == pytest.approx(-0.8824, abs=0.0001)
    assert test_greeks.gamma == pytest.approx(0, abs=0.0001)
    assert test_greeks.vega == pytest.approx(0, abs=0.0001)
    assert test_greeks.theta == pytest.approx(-2.5098, abs=0.0001)
    assert test_greeks.rho == pytest.approx(-237.9311, abs=0.0001)    