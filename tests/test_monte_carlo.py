import pytest 
from datetime import date, timedelta
import math

from core.enums import OptionType
from instruments.european import EuropeanOption
from market.environment import MarketEnvironment

from engines.monte_carlo import MonteCarloPricer
from engines.black_scholes_merton import BlackScholesMertonPricer # for comparisons

#----------------------------- Option values test - known values
def test_call_pricing_hull():
    test_market = MarketEnvironment(spot_price=50, risk_free_rate=0.05, volatility=0.3, dividend_yield=0.0, pricing_date=date(2026, 3, 18))
    test_call_option = EuropeanOption(strike_price=50, expiry_date=date(2026, 9, 18), option_type=OptionType.CALL)
    mc_price = test_call_option.price(pricer=MonteCarloPricer(num_paths=1000, random_seed=42), market_env=test_market)
    bsm_price = test_call_option.price(pricer=BlackScholesMertonPricer(), market_env=test_market)

    assert mc_price == pytest.approx(bsm_price, abs=0.5) # mc_price within (bsm_price - 0.5, bsm_price + 0.5) 

def test_put_against_bsm():
    test_market = MarketEnvironment(spot_price=50, risk_free_rate=0.05, volatility=0.3, dividend_yield=0.0, pricing_date=date(2026, 3, 18))
    test_put_option = EuropeanOption(strike_price=50, expiry_date=date(2026, 9, 18), option_type=OptionType.PUT)
    mc_price = test_put_option.price(pricer=MonteCarloPricer(num_paths=1000, random_seed=42), market_env=test_market)
    bsm_price = test_put_option.price(pricer=BlackScholesMertonPricer(), market_env=test_market)

    assert mc_price == pytest.approx(bsm_price, abs=0.5)

#----------------------------- Put-Call parity test (with dividend yield)
def test_put_call_parity():
    test_market = MarketEnvironment(spot_price=100, risk_free_rate=0.05, volatility=0.25, dividend_yield=0.025, pricing_date=date(2026, 3, 18))
    test_call_option = EuropeanOption(strike_price=90, expiry_date=date(2027, 4, 28), option_type=OptionType.CALL)
    test_put_option = EuropeanOption(strike_price=90, expiry_date=date(2027, 4, 28), option_type=OptionType.PUT)

    price_call = test_call_option.price(pricer=MonteCarloPricer(num_paths=10000, random_seed=42), market_env=test_market)
    price_put = test_put_option.price(pricer=MonteCarloPricer(num_paths=10000, random_seed=124), market_env=test_market) # using different random numbers
    T_minus_t = test_call_option.time_to_maturity(test_market.pricing_date)

    assert (price_call - price_put) == pytest.approx(test_market.spot_price * math.exp(-test_market.dividend_yield * T_minus_t) - test_call_option.strike_price * math.exp(-test_market.risk_free_rate * T_minus_t), abs=0.5)     

#----------------------------- Convergence to BSM values test
def test_converging_to_BSM_call():
    test_market = MarketEnvironment(spot_price=80, risk_free_rate=-0.015, volatility=0.4, dividend_yield=0.05, pricing_date=date(2026, 3, 18))
    test_call_option = EuropeanOption(strike_price=80, expiry_date=test_market.pricing_date + timedelta(days=21/12*365), option_type=OptionType.CALL) # option expires in 21 months

    price_call_mc = test_call_option.price(pricer=MonteCarloPricer(num_paths=600000, random_seed=42), market_env=test_market)
    price_call_BSM = test_call_option.price(pricer=BlackScholesMertonPricer(), market_env=test_market)

    assert price_call_mc == pytest.approx(price_call_BSM, abs=0.01)

def test_converging_to_BSM_put():
    test_market = MarketEnvironment(spot_price=150, risk_free_rate=0.03, volatility=0.6, dividend_yield=0.05, pricing_date=date(2026, 3, 18))
    test_put_option = EuropeanOption(strike_price=180, expiry_date=test_market.pricing_date + timedelta(days=7/12*365), option_type=OptionType.PUT) # option expires in 7 months

    price_put_mc = test_put_option.price(pricer=MonteCarloPricer(num_paths=500000, random_seed=142), market_env=test_market)
    price_put_BSM = test_put_option.price(pricer=BlackScholesMertonPricer(), market_env=test_market)

    assert price_put_mc == pytest.approx(price_put_BSM, abs=0.01) 

#----------------------------- Greek values test - comparing numerical Greeks to BSM Greeks which were already tested against known values in test_black_scholes_merton.py
def test_call_Greeks_against_BSM():
    test_market = MarketEnvironment(spot_price=100, risk_free_rate=0.02, volatility=0.4, dividend_yield=0.05, pricing_date=date(2026, 3, 18))
    test_call_option = EuropeanOption(strike_price=100, expiry_date=date(2026, 9, 16), option_type=OptionType.CALL)
    test_greeks_BSM = test_call_option.greeks(pricer=BlackScholesMertonPricer(), market_env=test_market) # Compute the Greeks for the test using BSM
    test_greeks_mc = test_call_option.greeks(pricer=MonteCarloPricer(num_paths=500000, random_seed=142), market_env=test_market) # Compute the Greeks for the test using the numerical method

    assert test_greeks_BSM.delta == pytest.approx(test_greeks_mc.delta, abs=0.01)
    assert test_greeks_BSM.gamma == pytest.approx(test_greeks_mc.gamma, abs=0.01)
    assert test_greeks_BSM.vega == pytest.approx(test_greeks_mc.vega, abs=0.1)
    assert test_greeks_BSM.theta == pytest.approx(test_greeks_mc.theta, abs=0.05)
    assert test_greeks_BSM.rho == pytest.approx(test_greeks_mc.rho, abs=0.05)  

#----------------------------- Reproducibility tests
def test_put_reproducible_mc():
    test_market = MarketEnvironment(spot_price=100, risk_free_rate=0.05, volatility=0.25, dividend_yield=0.025, pricing_date=date(2026, 3, 18))
    test_put_option = EuropeanOption(strike_price=90, expiry_date=date(2027, 3, 18), option_type=OptionType.PUT)  

    mc_price_seed142_a = test_put_option.price(pricer=MonteCarloPricer(num_paths=1000, random_seed=142), market_env=test_market) 
    mc_price_seed142_b = test_put_option.price(pricer=MonteCarloPricer(num_paths=1000, random_seed=142), market_env=test_market) 
    mc_price_diff_seed = test_put_option.price(pricer=MonteCarloPricer(num_paths=1000, random_seed=20), market_env=test_market) 

    assert mc_price_seed142_a == mc_price_seed142_b
    assert mc_price_seed142_a != mc_price_diff_seed

