import pytest 
from datetime import date, timedelta
import math

from core.enums import OptionType, AsianType
from instruments.european import EuropeanOption
from instruments.asian import AsianOption
from market.environment import MarketEnvironment

from engines.monte_carlo import MonteCarloPricer
from engines.black_scholes_merton import BlackScholesMertonPricer # for comparisons

#----------------------------- Define fixtures
# fixtures also in conftest.py
#--------------- MarketEnvironment instances
@pytest.fixture
def market_env_4():
    return MarketEnvironment(spot_price=50, risk_free_rate=0.05, volatility=0.3, dividend_yield=0.0, pricing_date=date(2026, 3, 18))

#--------------- EuropeanOption or AsianOption instances
@pytest.fixture
def asian_call_fixed_strike_1():
    return AsianOption(strike_price=100, expiry_date=date(2027, 2, 14), option_type=OptionType.CALL, asian_type=AsianType.FIXED_STRIKE)

@pytest.fixture
def european_call_option_5():
    return EuropeanOption(strike_price=100, expiry_date=date(2027, 2, 14), option_type=OptionType.CALL)

#----------------------------- Option values test - known values (European options)
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

#----------------------------- Option values test - known values (Asian options) 
# reference prices obtained from the online calculator available at: https://www.coggit.com/tools/arithmetic_asian_option_prices.html   
def test_asian_call():
    test_market = MarketEnvironment(spot_price=100, risk_free_rate=0.05, volatility=0.4, dividend_yield=0.025, pricing_date=date(2026, 4, 6))
    asian_call_option = AsianOption(strike_price=100, expiry_date=test_market.pricing_date + timedelta(days=365), option_type=OptionType.CALL, asian_type=AsianType.FIXED_STRIKE)

    asian_call_price = asian_call_option.price(pricer=MonteCarloPricer(num_paths=100000, num_steps=4, random_seed=42), market_env=test_market)
    assert asian_call_price == pytest.approx(9.035, abs=0.1) # 9.035 is halfway between the analytical approximations from the calculator

def test_asian_put():
    test_market = MarketEnvironment(spot_price=90, risk_free_rate=0.10, volatility=0.2, dividend_yield=0.02, pricing_date=date(2026, 4, 6))
    asian_put_option = AsianOption(strike_price=100, expiry_date=test_market.pricing_date + timedelta(days=365), option_type=OptionType.PUT, asian_type=AsianType.FIXED_STRIKE)

    asian_put_price = asian_put_option.price(pricer=MonteCarloPricer(num_paths=100000, num_steps=99, random_seed=42), market_env=test_market)
    assert asian_put_price == pytest.approx(7.55, abs=0.1) # 7.55 is halfway between the analytical approximations from the calculator   

#----------------------------- Fixed-strike Asian options less or equal to European options 
def test_Asian_vs_European_calls(asian_call_fixed_strike_1, european_call_option_5, market_env_3):
    asian_call_price = asian_call_fixed_strike_1.price(pricer=MonteCarloPricer(num_paths=100000, num_steps=50, random_seed=22), market_env=market_env_3)
    european_call_price = european_call_option_5.price(pricer=MonteCarloPricer(num_paths=100000, num_steps=50, random_seed=1112), market_env=market_env_3)

    assert asian_call_price <= european_call_price

def test_Asian_vs_European_puts(european_put_option_3, market_env_1):
    asian_put = AsianOption(strike_price=90, expiry_date=date(2027, 4, 28), option_type=OptionType.PUT, asian_type=AsianType.FIXED_STRIKE)  # same parameters as european_put_option_3
    asian_put_price = asian_put.price(pricer=MonteCarloPricer(num_paths=100000, num_steps=100, random_seed=1), market_env=market_env_1)
    european_put_price = european_put_option_3.price(pricer=MonteCarloPricer(num_paths=100000, num_steps=100, random_seed=111), market_env=market_env_1)

    assert asian_put_price <= european_put_price



#----------------------------- Floating-strike Asian prices are positive and finite 
@pytest.mark.parametrize(
    "spot, rate, div, vol, T",[
        (200.0, 0.10, 0.05, 0.35, date(2027, 1, 28)), # Standard
        (120.0, 0.05, 0.00, 0.20, date(2026, 4, 7)), # Zero dividend yield
        (80.0, 0.00, 0.08, 0.50, date(2026, 10, 20)), # High Div, High Vol
        (100.0, 0.00, 0.00, 0.10, date(2026, 4, 28)), # Zero rates/div, Low Vol
        (100.0, 0.20, 0.00, 0.80, date(2028, 4, 6)), # High Interest Rates, High Vol
    ]
)  
def test_floating_strike_call_asians(spot, rate, div, vol, T):
    test_market = MarketEnvironment(spot_price=spot, risk_free_rate=rate, volatility=vol, dividend_yield=div, pricing_date=date(2026, 4, 6)) 
    asian_call_floating = AsianOption(expiry_date=T, option_type=OptionType.CALL, asian_type=AsianType.FLOATING_STRIKE)
    asian_put_floating = AsianOption(expiry_date=T, option_type=OptionType.PUT, asian_type=AsianType.FLOATING_STRIKE)

    asian_call_price = asian_call_floating.price(pricer=MonteCarloPricer(num_paths=1000, num_steps=100, random_seed=132), market_env=test_market)
    asian_put_price = asian_put_floating.price(pricer=MonteCarloPricer(num_paths=1000, num_steps=100, random_seed=138), market_env=test_market)

    assert math.isfinite(asian_call_price)
    assert math.isfinite(asian_put_price)
    assert asian_call_price >= 0
    assert asian_put_price >= 0  
    assert asian_call_price < spot
    assert asian_put_price < spot


#----------------------------- Put-Call parity test with dividend yield (European options)
def test_put_call_parity(market_env_1, european_call_option_3, european_put_option_3):
    price_call = european_call_option_3.price(pricer=MonteCarloPricer(num_paths=10000, random_seed=42), market_env=market_env_1)
    price_put = european_put_option_3.price(pricer=MonteCarloPricer(num_paths=10000, random_seed=124), market_env=market_env_1) # using different random numbers
    T_minus_t = european_call_option_3.time_to_maturity(market_env_1.pricing_date)

    assert (price_call - price_put) == pytest.approx(market_env_1.spot_price * math.exp(-market_env_1.dividend_yield * T_minus_t) - european_call_option_3.strike_price * math.exp(-market_env_1.risk_free_rate * T_minus_t), abs=0.5)     

#----------------------------- Convergence to BSM values test (European options)
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

#----------------------------- Greek values test - comparing numerical Greeks to BSM Greeks which were already tested against known values in test_black_scholes_merton.py (European options)
def test_call_Greeks_against_BSM(market_env_3, european_call_option_4):
    test_greeks_BSM = european_call_option_4.greeks(pricer=BlackScholesMertonPricer(), market_env=market_env_3) # Compute the Greeks for the test using BSM
    test_greeks_MC = european_call_option_4.greeks(pricer=MonteCarloPricer(num_paths=500000, random_seed=142), market_env=market_env_3) # Compute the Greeks for the test using the numerical method

    assert test_greeks_BSM.delta == pytest.approx(test_greeks_MC.delta, abs=0.01)
    assert test_greeks_BSM.gamma == pytest.approx(test_greeks_MC.gamma, abs=0.01)
    assert test_greeks_BSM.vega == pytest.approx(test_greeks_MC.vega, abs=0.1)
    assert test_greeks_BSM.theta == pytest.approx(test_greeks_MC.theta, abs=0.05)
    assert test_greeks_BSM.rho == pytest.approx(test_greeks_MC.rho, abs=0.05) 

#----------------------------- Greek values test - Asian options 
@pytest.mark.parametrize(
    "strike, spot, rate, div, vol, T",[
        (200.0, 200.0, 0.10, 0.05, 0.35, date(2027, 1, 28)), # Standard
        (130.0, 120.0, 0.05, 0.00, 0.20, date(2026, 8, 7)), # Zero dividend yield
        (90.0, 80.0, 0.00, 0.08, 0.50, date(2026, 10, 20)), # High Div, High Vol
        (100.0, 100.0, 0.00, 0.00, 0.10, date(2026, 4, 28)), # Zero rates/div, Low Vol
        (100.0, 100.0, 0.20, 0.00, 0.80, date(2028, 4, 6)), # High Interest Rates, High Vol
    ]
)  
def test_Greeks_asians(strike, spot, rate, div, vol, T):
    test_market = MarketEnvironment(spot_price=spot, risk_free_rate=rate, volatility=vol, dividend_yield=div, pricing_date=date(2026, 4, 6)) 
    
    asian_call_floating = AsianOption(expiry_date=T, option_type=OptionType.CALL, asian_type=AsianType.FLOATING_STRIKE)
    asian_put_floating = AsianOption(expiry_date=T, option_type=OptionType.PUT, asian_type=AsianType.FLOATING_STRIKE)
    
    asian_call_fixed = AsianOption(strike_price=strike, expiry_date=T, option_type=OptionType.CALL, asian_type=AsianType.FIXED_STRIKE)
    asian_put_fixed = AsianOption(strike_price=strike, expiry_date=T, option_type=OptionType.PUT, asian_type=AsianType.FIXED_STRIKE)

    european_call = EuropeanOption(strike_price=strike, expiry_date=T, option_type=OptionType.CALL)
    european_put = EuropeanOption(strike_price=strike, expiry_date=T, option_type=OptionType.PUT)

    floating_strike_call_Greeks = asian_call_floating.greeks(pricer=MonteCarloPricer(num_paths=1000, num_steps=50, random_seed=58), market_env=test_market)
    floating_strike_put_Greeks = asian_put_floating.greeks(pricer=MonteCarloPricer(num_paths=1000, num_steps=50, random_seed=162), market_env=test_market)

    fixed_strike_call_Greeks = asian_call_fixed.greeks(pricer=MonteCarloPricer(num_paths=1000, num_steps=50, random_seed=25), market_env=test_market)
    fixed_strike_put_Greeks = asian_put_fixed.greeks(pricer=MonteCarloPricer(num_paths=1000, num_steps=50, random_seed=18), market_env=test_market)

    european_call_Greeks = european_call.greeks(pricer=BlackScholesMertonPricer(), market_env=test_market)
    european_put_Greeks = european_put.greeks(pricer=BlackScholesMertonPricer(), market_env=test_market)

    # Delta checks
    assert 0.0 <= floating_strike_call_Greeks.delta <= 1.0
    assert 0.0 <= floating_strike_put_Greeks.delta <= 1.0 # Floating Strike Puts have POSITIVE delta (Delta = V / S_0)

    assert 0.0 <= fixed_strike_call_Greeks.delta <= 1.0
    assert -1.0 <= fixed_strike_put_Greeks.delta <= 0.0

    # fixed-strike Asian vega should be smaller in magnitude than European vega (because averaging dampens sensitivity)
    # these tests could fail in extreme market environments
    assert fixed_strike_call_Greeks.vega < european_call_Greeks.vega
    assert fixed_strike_put_Greeks.vega < european_put_Greeks.vega
    
    TOLERANCE = -1e-10 # for floating-point comparisons
    # Gamma positive
    assert floating_strike_call_Greeks.gamma >= TOLERANCE
    assert floating_strike_put_Greeks.gamma >= TOLERANCE

    assert fixed_strike_call_Greeks.gamma >= TOLERANCE
    assert fixed_strike_put_Greeks.gamma >= TOLERANCE
    
    # Vega positive
    assert floating_strike_call_Greeks.vega >= TOLERANCE
    assert floating_strike_put_Greeks.vega >= TOLERANCE

    assert fixed_strike_call_Greeks.vega >= TOLERANCE
    assert fixed_strike_put_Greeks.vega >= TOLERANCE     

#----------------------------- Reproducibility tests for Monte Carlo (European options)
def test_put_reproducible_MC(market_env_1, european_put_option_1):
    MC_price_seed142_a = european_put_option_1.price(pricer=MonteCarloPricer(num_paths=1000, random_seed=142), market_env=market_env_1) 
    MC_price_seed142_b = european_put_option_1.price(pricer=MonteCarloPricer(num_paths=1000, random_seed=142), market_env=market_env_1) 
    MC_price_diff_seed = european_put_option_1.price(pricer=MonteCarloPricer(num_paths=1000, random_seed=20), market_env=market_env_1) 

    assert MC_price_seed142_a == MC_price_seed142_b
    assert MC_price_seed142_a != MC_price_diff_seed

#----------------------------- Reproducibility tests for Monte Carlo (Asian options)
def test_Asian_call_reproducible_MC(market_env_1, asian_call_fixed_strike_1):
    MC_price_seed80_a = asian_call_fixed_strike_1.price(pricer=MonteCarloPricer(num_paths=1000, random_seed=80), market_env=market_env_1) 
    MC_price_seed80_b = asian_call_fixed_strike_1.price(pricer=MonteCarloPricer(num_paths=1000, random_seed=80), market_env=market_env_1) 
    MC_price_diff_seed = asian_call_fixed_strike_1.price(pricer=MonteCarloPricer(num_paths=1000, random_seed=12), market_env=market_env_1) 

    assert MC_price_seed80_a == MC_price_seed80_b
    assert MC_price_seed80_a != MC_price_diff_seed