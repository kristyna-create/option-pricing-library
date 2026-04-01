import pytest 
from datetime import date, timedelta
import math

from core.enums import OptionType
from instruments.european import EuropeanOption
from market.environment import MarketEnvironment

from engines.monte_carlo import MonteCarloPricer
from engines.black_scholes_merton import BlackScholesMertonPricer # for comparisons

#----------------------------- Define fixtures
# fixtures also in conftest.py
#--------------- MarketEnvironment instances
@pytest.fixture
def market_env_4():
    return MarketEnvironment(spot_price=50, risk_free_rate=0.05, volatility=0.3, dividend_yield=0.0, pricing_date=date(2026, 3, 18))

#--------------- EuropeanOption instances

#----------------------------- Option values test - known values
def test_call_pricing_hull(market_env_4):
    test_call_option = EuropeanOption(strike_price=50, expiry_date=date(2026, 9, 18), option_type=OptionType.CALL) # unique, fixture not defined
    MC_price = test_call_option.price(pricer=MonteCarloPricer(num_paths=1000, random_seed=42), market_env=market_env_4)
    bsm_price = test_call_option.price(pricer=BlackScholesMertonPricer(), market_env=market_env_4)

    assert MC_price == pytest.approx(bsm_price, abs=0.5) # MC_price within (bsm_price - 0.5, bsm_price + 0.5) 

def test_put_against_bsm(market_env_4):
    test_put_option = EuropeanOption(strike_price=50, expiry_date=date(2026, 9, 18), option_type=OptionType.PUT) # unique, fixture not defined
    MC_price = test_put_option.price(pricer=MonteCarloPricer(num_paths=1000, random_seed=42), market_env=market_env_4)
    bsm_price = test_put_option.price(pricer=BlackScholesMertonPricer(), market_env=market_env_4)

    assert MC_price == pytest.approx(bsm_price, abs=0.5)

#----------------------------- Put-Call parity test (with dividend yield)
def test_put_call_parity(market_env_1, european_call_option_3, european_put_option_3):
    price_call = european_call_option_3.price(pricer=MonteCarloPricer(num_paths=10000, random_seed=42), market_env=market_env_1)
    price_put = european_put_option_3.price(pricer=MonteCarloPricer(num_paths=10000, random_seed=124), market_env=market_env_1) # using different random numbers
    T_minus_t = european_call_option_3.time_to_maturity(market_env_1.pricing_date)

    assert (price_call - price_put) == pytest.approx(market_env_1.spot_price * math.exp(-market_env_1.dividend_yield * T_minus_t) - european_call_option_3.strike_price * math.exp(-market_env_1.risk_free_rate * T_minus_t), abs=0.5)     

#----------------------------- Convergence to BSM values test
def test_converging_to_BSM_call():
    test_market = MarketEnvironment(spot_price=80, risk_free_rate=-0.015, volatility=0.4, dividend_yield=0.05, pricing_date=date(2026, 3, 18)) # unique, fixture not defined
    test_call_option = EuropeanOption(strike_price=80, expiry_date=test_market.pricing_date + timedelta(days=21/12*365), option_type=OptionType.CALL) # option expires in 21 months

    price_call_MC = test_call_option.price(pricer=MonteCarloPricer(num_paths=600000, random_seed=42), market_env=test_market)
    price_call_BSM = test_call_option.price(pricer=BlackScholesMertonPricer(), market_env=test_market)

    assert price_call_MC == pytest.approx(price_call_BSM, abs=0.01)

def test_converging_to_BSM_put():
    test_market = MarketEnvironment(spot_price=150, risk_free_rate=0.03, volatility=0.6, dividend_yield=0.05, pricing_date=date(2026, 3, 18)) # unique, fixture not defined
    test_put_option = EuropeanOption(strike_price=180, expiry_date=test_market.pricing_date + timedelta(days=7/12*365), option_type=OptionType.PUT) # option expires in 7 months

    price_put_MC = test_put_option.price(pricer=MonteCarloPricer(num_paths=500000, random_seed=142), market_env=test_market)
    price_put_BSM = test_put_option.price(pricer=BlackScholesMertonPricer(), market_env=test_market)

    assert price_put_MC == pytest.approx(price_put_BSM, abs=0.01) 

#----------------------------- Greek values test - comparing numerical Greeks to BSM Greeks which were already tested against known values in test_black_scholes_merton.py
def test_call_Greeks_against_BSM(market_env_3, european_call_option_4):
    test_greeks_BSM = european_call_option_4.greeks(pricer=BlackScholesMertonPricer(), market_env=market_env_3) # Compute the Greeks for the test using BSM
    test_greeks_MC = european_call_option_4.greeks(pricer=MonteCarloPricer(num_paths=500000, random_seed=142), market_env=market_env_3) # Compute the Greeks for the test using the numerical method

    assert test_greeks_BSM.delta == pytest.approx(test_greeks_MC.delta, abs=0.01)
    assert test_greeks_BSM.gamma == pytest.approx(test_greeks_MC.gamma, abs=0.01)
    assert test_greeks_BSM.vega == pytest.approx(test_greeks_MC.vega, abs=0.1)
    assert test_greeks_BSM.theta == pytest.approx(test_greeks_MC.theta, abs=0.05)
    assert test_greeks_BSM.rho == pytest.approx(test_greeks_MC.rho, abs=0.05)  

#----------------------------- Reproducibility tests for Monte Carlo
def test_put_reproducible_MC(market_env_1, european_put_option_1):
    MC_price_seed142_a = european_put_option_1.price(pricer=MonteCarloPricer(num_paths=1000, random_seed=142), market_env=market_env_1) 
    MC_price_seed142_b = european_put_option_1.price(pricer=MonteCarloPricer(num_paths=1000, random_seed=142), market_env=market_env_1) 
    MC_price_diff_seed = european_put_option_1.price(pricer=MonteCarloPricer(num_paths=1000, random_seed=20), market_env=market_env_1) 

    assert MC_price_seed142_a == MC_price_seed142_b
    assert MC_price_seed142_a != MC_price_diff_seed

