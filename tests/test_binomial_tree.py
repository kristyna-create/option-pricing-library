import pytest 
from datetime import date, timedelta
import math

from core.enums import OptionType
from instruments.european import EuropeanOption
from market.environment import MarketEnvironment

from engines.binomial_tree import BinomialTreePricer
from engines.black_scholes_merton import BlackScholesMertonPricer # for comparisons

#----------------------------- Define fixtures
# fixtures also in conftest.py

#----------------------------- Option values test - known values
# this example is from Hull, page 485, Figure 21.10
def test_put_pricing_hull():
    test_market = MarketEnvironment(spot_price=50, risk_free_rate=0.10, volatility=0.4, dividend_yield=0.0, pricing_date=date(2026, 3, 18)) # unique, fixture not defined
    test_put_option = EuropeanOption(strike_price=50, expiry_date=test_market.pricing_date + timedelta(days=5/12*365), option_type=OptionType.PUT) # unique, fixture not defined

    assert test_put_option.price(pricer=BinomialTreePricer(num_tree_steps=5), market_env=test_market) == pytest.approx(4.32, abs=0.01)

# known values below are obtained from the online calculator available at: https://www.neurolab.de/cgi-bin/binomial.cgi 
def test_call_pricing_known_value(european_call_option_1):
    test_market = MarketEnvironment(spot_price=100, risk_free_rate=0.05, volatility=0.3, dividend_yield=0.0, pricing_date=date(2026, 3, 18)) # unique, fixture not defined

    assert european_call_option_1.price(pricer=BinomialTreePricer(num_tree_steps=4), market_env=test_market) == pytest.approx(13.524, abs=0.0001)   
 
def test_put_pricing_known_value():
    test_market = MarketEnvironment(spot_price=100, risk_free_rate=0.05, volatility=0.4, dividend_yield=0.0, pricing_date=date(2026, 3, 18)) # unique, fixture not defined
    test_put_option = EuropeanOption(strike_price=100, expiry_date=test_market.pricing_date + timedelta(days=1.25*365), option_type=OptionType.PUT) # to match 1.25 years to maturity

    assert test_put_option.price(pricer=BinomialTreePricer(num_tree_steps=10), market_env=test_market) == pytest.approx(13.866159, abs=0.01)  

#----------------------------- Put-Call parity test (with dividend yield)
def test_put_call_parity(market_env_1, european_call_option_3, european_put_option_3):
    price_call = european_call_option_3.price(pricer=BinomialTreePricer(num_tree_steps=100), market_env=market_env_1)
    price_put = european_put_option_3.price(pricer=BinomialTreePricer(num_tree_steps=100), market_env=market_env_1)
    T_minus_t = european_call_option_3.time_to_maturity(market_env_1.pricing_date)

    assert (price_call - price_put) == pytest.approx(market_env_1.spot_price * math.exp(-market_env_1.dividend_yield * T_minus_t) - european_call_option_3.strike_price * math.exp(-market_env_1.risk_free_rate * T_minus_t), abs=0.0001) 

#----------------------------- Convergence to BSM values test
def test_converging_to_BSM_call():
    test_market = MarketEnvironment(spot_price=200, risk_free_rate=-0.015, volatility=0.4, dividend_yield=0.05, pricing_date=date(2026, 3, 18)) # unique, fixture not defined
    test_call_option = EuropeanOption(strike_price=210, expiry_date=test_market.pricing_date + timedelta(days=19/12*365), option_type=OptionType.CALL) # option expires in 19 months

    price_call_tree = test_call_option.price(pricer=BinomialTreePricer(num_tree_steps=1000), market_env=test_market)
    price_call_BSM = test_call_option.price(pricer=BlackScholesMertonPricer(), market_env=test_market)

    assert price_call_tree == pytest.approx(price_call_BSM, abs=0.01)

def test_converging_to_BSM_put():
    test_market = MarketEnvironment(spot_price=150, risk_free_rate=0.10, volatility=0.4, dividend_yield=0.05, pricing_date=date(2026, 3, 18)) # unique, fixture not defined
    test_put_option = EuropeanOption(strike_price=150, expiry_date=test_market.pricing_date + timedelta(days=5/12*365), option_type=OptionType.PUT) # option expires in 5 months

    price_put_tree = test_put_option.price(pricer=BinomialTreePricer(num_tree_steps=1000), market_env=test_market)
    price_put_BSM = test_put_option.price(pricer=BlackScholesMertonPricer(), market_env=test_market)

    assert price_put_tree == pytest.approx(price_put_BSM, abs=0.01)


#----------------------------- Greek values test - comparing numerical Greeks to BSM Greeks which were already tested against known values in test_black_scholes_merton.py
def test_call_Greeks_against_BSM(market_env_3, european_call_option_4):
    test_greeks_BSM = european_call_option_4.greeks(pricer=BlackScholesMertonPricer(), market_env=market_env_3) # Compute the Greeks for the test using BSM
    test_greeks_tree = european_call_option_4.greeks(pricer=BinomialTreePricer(num_tree_steps=5000), market_env=market_env_3) # Compute the Greeks for the test using the numerical method

    assert test_greeks_BSM.delta == pytest.approx(test_greeks_tree.delta, abs=0.01)
    assert test_greeks_BSM.gamma == pytest.approx(test_greeks_tree.gamma, abs=0.01)
    assert test_greeks_BSM.vega == pytest.approx(test_greeks_tree.vega, abs=0.01)
    assert test_greeks_BSM.theta == pytest.approx(test_greeks_tree.theta, abs=0.1)
    assert test_greeks_BSM.rho == pytest.approx(test_greeks_tree.rho, abs=0.01)
    # gamma from the tree with num_tree_steps=1000 are below, BSM gamma gives 0.0137, so should be close to this:
    # 0.0245 with DELTA_BUMP_PCT = 0.01
    # 0.049 with DELTA_BUMP_PCT = 0.005
    # 0.245 with DELTA_BUMP_PCT = 0.001

