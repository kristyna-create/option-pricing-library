import pytest 
from datetime import date, timedelta
import math

from core.enums import OptionType
from instruments.european import EuropeanOption
from instruments.american import AmericanOption
from market.environment import MarketEnvironment

from engines.binomial_tree import BinomialTreePricer
from engines.black_scholes_merton import BlackScholesMertonPricer # for comparisons

#----------------------------- Define fixtures
# fixtures also in conftest.py

#----------------------------- Option values test - known values (European options)
# this example is from Hull, page 485, Figure 21.10
def test_EU_put_pricing_hull():
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

#----------------------------- Option values test - known values (American options)
# this is Example 21.1 from Hull, page 473 
def test_AM_put_pricing_hull():
    test_market = MarketEnvironment(spot_price=50, risk_free_rate=0.10, volatility=0.40, pricing_date=date(2026, 4, 3))
    AM_put_option = AmericanOption(strike_price=50, expiry_date=test_market.pricing_date + timedelta(5/12 * 365), option_type=OptionType.PUT) # option expires in 5 months

    assert AM_put_option.price(pricer=BinomialTreePricer(num_tree_steps=5), market_env=test_market) == pytest.approx(4.49, abs=0.01)

# Hull Example 21.1, option values from DerivaGem
@pytest.mark.parametrize("tree_steps, option_value", [
    (30, 4.263),
    (50, 4.272),
    (100, 4.278),
    (500, 4.283)
])
def test_AM_put_hull(tree_steps, option_value):
    test_market = MarketEnvironment(spot_price=50, risk_free_rate=0.10, volatility=0.40, pricing_date=date(2026, 4, 3))
    AM_put_option = AmericanOption(strike_price=50, expiry_date=test_market.pricing_date + timedelta(5/12 * 365), option_type=OptionType.PUT) # option expires in 5 months

    assert AM_put_option.price(pricer=BinomialTreePricer(num_tree_steps=tree_steps), market_env=test_market) == pytest.approx(option_value, abs=0.001)

# this is Example 21.5 from Hull, page 478, option values produced by DerivaGem
@pytest.mark.parametrize("tree_steps, option_value", [
    (4, 19.16),
    (50, 20.18),
    (100, 20.22)
])
def test_AM_call_hull(tree_steps, option_value):
    test_market = MarketEnvironment(spot_price=300, risk_free_rate=0.08, dividend_yield=0.08, volatility=0.30, pricing_date=date(2026, 4, 3))
    AM_call_option = AmericanOption(strike_price=300, expiry_date=date(2026, 8, 3), option_type=OptionType.CALL)

    assert AM_call_option.price(pricer=BinomialTreePricer(num_tree_steps=tree_steps), market_env=test_market) == pytest.approx(option_value, abs=0.05)
      
#----------------------------- Put-Call parity test (with dividend yield) (European options)
def test_put_call_parity(market_env_1, european_call_option_3, european_put_option_3):
    price_call = european_call_option_3.price(pricer=BinomialTreePricer(num_tree_steps=100), market_env=market_env_1)
    price_put = european_put_option_3.price(pricer=BinomialTreePricer(num_tree_steps=100), market_env=market_env_1)
    T_minus_t = european_call_option_3.time_to_maturity(market_env_1.pricing_date)

    assert (price_call - price_put) == pytest.approx(market_env_1.spot_price * math.exp(-market_env_1.dividend_yield * T_minus_t) - european_call_option_3.strike_price * math.exp(-market_env_1.risk_free_rate * T_minus_t), abs=0.0001) 

#----------------------------- Put-Call parity test for non-dividend-paying stocks (American options)
def test_AM_put_call_parity():
    strike = 200
    test_market = MarketEnvironment(spot_price=200, risk_free_rate=0.10, volatility=0.40, pricing_date=date(2026, 4, 3))
    AM_call = AmericanOption(strike_price=strike, expiry_date=date(2026, 10, 3), option_type=OptionType.CALL)
    AM_put = AmericanOption(strike_price=strike, expiry_date=date(2026, 10, 3), option_type=OptionType.PUT)
    T_minus_t = AM_call.time_to_maturity(test_market.pricing_date)

    AM_call_price = AM_call.price(pricer=BinomialTreePricer(num_tree_steps=100), market_env=test_market)
    AM_put_price = AM_put.price(pricer=BinomialTreePricer(num_tree_steps=100), market_env=test_market)

    assert (test_market.spot_price - strike) <= (AM_call_price - AM_put_price) <= (test_market.spot_price - strike * math.exp(-test_market.risk_free_rate * T_minus_t))

#----------------------------- Convergence to BSM values test (European options)
def test_converging_to_BSM_call():
    test_market = MarketEnvironment(spot_price=200, risk_free_rate=-0.015, volatility=0.4, dividend_yield=0.05, pricing_date=date(2026, 3, 18)) # unique, fixture not defined
    test_call_option = EuropeanOption(strike_price=210, expiry_date=test_market.pricing_date + timedelta(days=19/12 * 365), option_type=OptionType.CALL) # option expires in 19 months

    price_call_tree = test_call_option.price(pricer=BinomialTreePricer(num_tree_steps=1000), market_env=test_market)
    price_call_BSM = test_call_option.price(pricer=BlackScholesMertonPricer(), market_env=test_market)

    assert price_call_tree == pytest.approx(price_call_BSM, abs=0.01)

def test_converging_to_BSM_put():
    test_market = MarketEnvironment(spot_price=150, risk_free_rate=0.10, volatility=0.4, dividend_yield=0.05, pricing_date=date(2026, 3, 18)) # unique, fixture not defined
    test_put_option = EuropeanOption(strike_price=150, expiry_date=test_market.pricing_date + timedelta(days=5/12*365), option_type=OptionType.PUT) # option expires in 5 months

    price_put_tree = test_put_option.price(pricer=BinomialTreePricer(num_tree_steps=1000), market_env=test_market)
    price_put_BSM = test_put_option.price(pricer=BlackScholesMertonPricer(), market_env=test_market)

    assert price_put_tree == pytest.approx(price_put_BSM, abs=0.01)


#----------------------------- Greek values test - comparing numerical Greeks to BSM Greeks which were already tested against known values in test_black_scholes_merton.py (European options)
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

#----------------------------- European vs American options price comparisons via Binomial Tree
# test that the price of an American call on a non-dividend-paying asset is equal to the European call (it is never optimal to exercise this call early)
def test_non_dividend_call_equality():
    test_market = MarketEnvironment(spot_price=100, risk_free_rate=0.05, volatility=0.25, dividend_yield=0.0, pricing_date=date(2026, 1, 25)) # unique, fixture not defined
    EU_call_option = EuropeanOption(strike_price=100, expiry_date=date(2026, 7, 25), option_type=OptionType.CALL) # unique, fixture not defined
    AM_call_option = AmericanOption(strike_price=100, expiry_date=date(2026, 7, 25), option_type=OptionType.CALL) # unique, fixture not defined

    EU_call_price = EU_call_option.price(pricer=BinomialTreePricer(num_tree_steps=1000), market_env=test_market)
    AM_call_price = AM_call_option.price(pricer=BinomialTreePricer(num_tree_steps=1000), market_env=test_market)

    assert EU_call_price == pytest.approx(AM_call_price, abs=1e-10)


# with dividends: American call price >= European call price   
def test_AM_vs_EU_call_dividends():
    test_market = MarketEnvironment(spot_price=100, risk_free_rate=0.05, volatility=0.4, dividend_yield=0.025, pricing_date=date(2026, 1, 25)) # unique, fixture not defined
    EU_call_option = EuropeanOption(strike_price=120, expiry_date=date(2026, 5, 25), option_type=OptionType.CALL) # unique, fixture not defined
    AM_call_option = AmericanOption(strike_price=120, expiry_date=date(2026, 5, 25), option_type=OptionType.CALL) # unique, fixture not defined

    EU_call_price = EU_call_option.price(pricer=BinomialTreePricer(num_tree_steps=1000), market_env=test_market)
    AM_call_price = AM_call_option.price(pricer=BinomialTreePricer(num_tree_steps=1000), market_env=test_market)

    assert AM_call_price >= EU_call_price

# for a put: always American put >= European put
def test_AM_vs_EU_put_no_dividends():
    test_market = MarketEnvironment(spot_price=100, risk_free_rate=0.08, volatility=0.3, dividend_yield=0.0, pricing_date=date(2026, 2, 14)) # unique, fixture not defined
    EU_put_option = EuropeanOption(strike_price=100, expiry_date=date(2026, 8, 14), option_type=OptionType.PUT) # unique, fixture not defined
    AM_put_option = AmericanOption(strike_price=100, expiry_date=date(2026, 8, 14), option_type=OptionType.PUT) # unique, fixture not defined

    EU_put_price = EU_put_option.price(pricer=BinomialTreePricer(num_tree_steps=1000), market_env=test_market)
    AM_put_price = AM_put_option.price(pricer=BinomialTreePricer(num_tree_steps=1000), market_env=test_market)

    assert AM_put_price >= EU_put_price

def test_AM_vs_EU_put_dividends():
    test_market = MarketEnvironment(spot_price=100, risk_free_rate=0.10, volatility=0.5, dividend_yield=0.05, pricing_date=date(2026, 4, 3)) # unique, fixture not defined
    EU_put_option = EuropeanOption(strike_price=90, expiry_date=date(2026, 9, 20), option_type=OptionType.PUT) # unique, fixture not defined
    AM_put_option = AmericanOption(strike_price=90, expiry_date=date(2026, 9, 20), option_type=OptionType.PUT) # unique, fixture not defined

    EU_put_price = EU_put_option.price(pricer=BinomialTreePricer(num_tree_steps=1000), market_env=test_market)
    AM_put_price = AM_put_option.price(pricer=BinomialTreePricer(num_tree_steps=1000), market_env=test_market)

    assert AM_put_price >= EU_put_price    

#----------------------------- Greek values test (American options)
# no closed-form solutions available to test against

# Greeks of American call should closely match European call Greeks with no dividends since their prices match
def test_AM_call_Greeks_against_BSM_no_dividends(european_call_option_4):
    test_market = MarketEnvironment(spot_price=100, risk_free_rate=0.10, volatility=0.35, pricing_date=date(2026, 4, 3))
    american_call_option = AmericanOption(strike_price=100, expiry_date=date(2026, 9, 16), option_type=OptionType.CALL) # matches european_call_option_4 from conftest.py
    test_greeks_BSM = european_call_option_4.greeks(pricer=BlackScholesMertonPricer(), market_env=test_market) # Compute the Greeks for the test using BSM
    test_greeks_tree = american_call_option.greeks(pricer=BinomialTreePricer(num_tree_steps=5000), market_env=test_market) # Compute the Greeks for the test using the numerical method

    assert test_greeks_BSM.delta == pytest.approx(test_greeks_tree.delta, abs=0.01)
    assert test_greeks_BSM.gamma == pytest.approx(test_greeks_tree.gamma, abs=0.01)
    assert test_greeks_BSM.vega == pytest.approx(test_greeks_tree.vega, abs=0.01)
    assert test_greeks_BSM.theta == pytest.approx(test_greeks_tree.theta, abs=0.1)
    assert test_greeks_BSM.rho == pytest.approx(test_greeks_tree.rho, abs=0.01)

# American call delta should be positive and between 0 and 1 
# American put delta should be negative and between -1 and 0
# Gamma should be positive for both American call and puts 
# Vega should be positive for both American call and puts     
@pytest.mark.parametrize(
    "spot, rate, div, vol",[
        (100.0, 0.10, 0.05, 0.35), # ATM, standard
        (120.0, 0.05, 0.00, 0.20), # ITM Call / OTM Put
        (80.0, 0.00, 0.08, 0.50), # OTM Call / ITM Put (High Div, High Vol)
        (100.0, 0.00, 0.00, 0.10), # ATM, Zero rates/div, Low Vol
        (100.0, 0.20, 0.00, 0.80), # High Interest Rates, High Vol
    ]
)
def test_AM_Greeks_sanity_checks(spot, rate, div, vol):
    test_market = MarketEnvironment(
        spot_price=spot, 
        risk_free_rate=rate, 
        dividend_yield=div, 
        volatility=vol, 
        pricing_date=date(2026, 4, 3)
    )

    american_call_option = AmericanOption(strike_price=100, expiry_date=date(2026, 10, 3), option_type=OptionType.CALL)
    american_put_option = AmericanOption(strike_price=100, expiry_date=date(2026, 10, 3), option_type=OptionType.PUT)

    pricer = BinomialTreePricer(num_tree_steps=1000)
    
    AM_call_greeks_tree = american_call_option.greeks(pricer=pricer, market_env=test_market)
    AM_put_greeks_tree = american_put_option.greeks(pricer=pricer, market_env=test_market)

    assert 0.0 <= AM_call_greeks_tree.delta <= 1.0
    assert -1.0 <= AM_put_greeks_tree.delta <= 0.0
    
    assert AM_call_greeks_tree.gamma >= 0.0
    assert AM_put_greeks_tree.gamma >= 0.0
    
    assert AM_call_greeks_tree.vega >= 0.0
    assert AM_put_greeks_tree.vega >= 0.0

      