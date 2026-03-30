import math
from scipy.stats import norm
import warnings

from core.pricer_base import BasePricer
from core.option_base import BaseOption
from instruments.european import EuropeanOption
from market.environment import MarketEnvironment
from core.enums import OptionType
from core.greeks_data import Greeks

class BlackScholesMertonPricer(BasePricer):

    TOLERANCE = 1e-10 # class-level constant for floating-point comparisons

    def _compute_d1_d2(self, option: BaseOption, market_env: MarketEnvironment, T_minus_t: float) -> tuple:
            d1 = (math.log(market_env.spot_price / option.strike_price) + (market_env.risk_free_rate - market_env.dividend_yield + 0.5 * market_env.volatility**2) * T_minus_t) / (market_env.volatility * math.sqrt(T_minus_t))

            d2 = d1 - market_env.volatility * math.sqrt(T_minus_t)

            return (d1, d2)
    
    # The core of this class - pricing method:
    def _calculate_price(self, option: BaseOption, market_env: MarketEnvironment) -> float:
        """Calculates the option price of EuropeanOption instances according to Black-Scholes-Merton."""
        if not isinstance(option, EuropeanOption):
            raise TypeError("BSM pricing is mathematically valid only for EuropeanOption instances!")
        
        T_minus_t = option.time_to_maturity(market_env.pricing_date)
        
        if abs(T_minus_t) < self.TOLERANCE: # Get option value at expiry
            return float(option.get_payoff(market_env.spot_price))
        elif market_env.volatility > self.TOLERANCE: # Normal case: some time remains to maturity and volatility is not zero 
            d1, d2 = self._compute_d1_d2(option, market_env, T_minus_t)

            if option.option_type == OptionType.CALL:
                price = market_env.spot_price * math.exp(-market_env.dividend_yield * T_minus_t) * float(norm.cdf(d1)) - option.strike_price * math.exp(-market_env.risk_free_rate * T_minus_t) * float(norm.cdf(d2))
            elif option.option_type == OptionType.PUT:
                price = -market_env.spot_price * math.exp(-market_env.dividend_yield * T_minus_t) * float(norm.cdf(-d1)) + option.strike_price * math.exp(-market_env.risk_free_rate * T_minus_t) * float(norm.cdf(-d2))
            else:
                valid_types = ", ".join(str(t) for t in OptionType)
                raise ValueError(f"Pricing EuropeanOption via BSM is currently implemented only for {valid_types} and you inserted {repr(option.option_type)}!")
            
            return price
        else: # Theoretical case option's value: when volatility is zero
            return float(math.exp(-market_env.risk_free_rate * T_minus_t) * option.get_payoff(market_env.spot_price * math.exp((market_env.risk_free_rate - market_env.dividend_yield) * T_minus_t)))

    # Closed-form Greeks for BSM - overrides _calculate_greeks() method from the parent (BasePricer) 
    def _calculate_greeks(self, option: BaseOption, market_env: MarketEnvironment) -> Greeks:
        """Calculates delta, gamma, vega, theta, rho of EuropeanOption instances according to Black-Scholes-Merton analytical formulas."""
        if not isinstance(option, EuropeanOption):
            raise TypeError("BSM pricing is mathematically valid only for EuropeanOption instances!")
        
        T_minus_t = option.time_to_maturity(market_env.pricing_date)

        if abs(T_minus_t) < self.TOLERANCE: # Greeks at expiry
            if option.option_type == OptionType.CALL:
                # at the money calls
                if abs(market_env.spot_price - option.strike_price) < self.TOLERANCE:
                    delta = 0.5
                    theta = float("-inf")
                    warnings.warn(f"Beware that for this case (ATM {option.option_type.value} at expiry), Theta approaches minus infinity!")
                # in the money calls
                elif market_env.spot_price > option.strike_price:
                    delta = 1.0
                    theta = market_env.dividend_yield * market_env.spot_price - market_env.risk_free_rate * option.strike_price
                # out of the money calls
                else:
                    delta = 0.0
                    theta = 0.0
            elif option.option_type == OptionType.PUT:
                # at the money puts    
                if abs(market_env.spot_price - option.strike_price) < self.TOLERANCE:    
                    delta = -0.5
                    theta = float("-inf")
                    warnings.warn(f"Beware that for this case (ATM {option.option_type.value} at expiry), Theta approaches minus infinity!")
                # out of the money puts
                elif market_env.spot_price > option.strike_price:
                    delta = 0.0
                    theta = 0.0
                # in the money puts    
                else:
                    delta = -1.0
                    theta = market_env.risk_free_rate * option.strike_price - market_env.dividend_yield * market_env.spot_price
            else:
                valid_types = ", ".join(str(t) for t in OptionType)
                raise ValueError(f"Pricing EuropeanOption via BSM is currently implemented only for {valid_types} and you inserted {repr(option.option_type)}!")

            # Vega and Rho for options at expiry:
            vega = 0.0
            rho = 0.0
            # Gamma for options at expiry:    
            if abs(market_env.spot_price - option.strike_price) < self.TOLERANCE: # at the money
                gamma = float("inf")
                warnings.warn(f"Beware that for this case (ATM {option.option_type.value} at expiry), Gamma approaches plus infinity!")
            else: # ITM or OTM options
                gamma = 0.0    

        elif market_env.volatility > self.TOLERANCE: # Normal case Greeks: some time remains to maturity and volatility is not zero
            d1, d2 = self._compute_d1_d2(option, market_env, T_minus_t)

            if option.option_type == OptionType.CALL:
                delta = math.exp(-market_env.dividend_yield * T_minus_t) * float(norm.cdf(d1))
                
                theta = -market_env.volatility * market_env.spot_price * math.exp(-market_env.dividend_yield * T_minus_t) * float(norm.pdf(d1)) / (2 * math.sqrt(T_minus_t)) + market_env.dividend_yield * market_env.spot_price * float(norm.cdf(d1)) * math.exp(-market_env.dividend_yield * T_minus_t) - market_env.risk_free_rate * option.strike_price * math.exp(-market_env.risk_free_rate * T_minus_t) * float(norm.cdf(d2))  
                
                rho = option.strike_price * T_minus_t * math.exp(-market_env.risk_free_rate * T_minus_t) * float(norm.cdf(d2))
            elif option.option_type == OptionType.PUT:
                delta = math.exp(-market_env.dividend_yield * T_minus_t) * (float(norm.cdf(d1)) - 1)
                
                theta = -market_env.volatility * market_env.spot_price * math.exp(-market_env.dividend_yield * T_minus_t) * float(norm.pdf(-d1)) / (2 * math.sqrt(T_minus_t)) - market_env.dividend_yield * market_env.spot_price * float(norm.cdf(-d1)) * math.exp(-market_env.dividend_yield * T_minus_t) + market_env.risk_free_rate * option.strike_price * math.exp(-market_env.risk_free_rate * T_minus_t) * float(norm.cdf(-d2)) 

                rho = -option.strike_price * T_minus_t * math.exp(-market_env.risk_free_rate * T_minus_t) * float(norm.cdf(-d2))
            else:
                valid_types = ", ".join(str(t) for t in OptionType)
                raise ValueError(f"Pricing EuropeanOption via BSM is currently implemented only for {valid_types} and you inserted {repr(option.option_type)}!")

            # Gamma and Vega are identical for European calls and puts
            gamma = math.exp(-market_env.dividend_yield * T_minus_t) * float(norm.pdf(d1)) / (market_env.volatility * market_env.spot_price * math.sqrt(T_minus_t))   
            vega = market_env.spot_price * math.sqrt(T_minus_t) * math.exp(-market_env.dividend_yield * T_minus_t) * float(norm.pdf(d1))
            
        else: # Theoretical case Greeks: when volatility is zero
            discounted_spot = market_env.spot_price * math.exp(-market_env.dividend_yield * T_minus_t)
            discounted_strike = option.strike_price * math.exp(-market_env.risk_free_rate * T_minus_t)

            if option.option_type == OptionType.CALL:
                # at the money calls
                if abs(discounted_spot - discounted_strike) < self.TOLERANCE:
                    delta = 0.5 * math.exp(-market_env.dividend_yield * T_minus_t)
                    theta = 0.5 * (market_env.dividend_yield * discounted_spot - market_env.risk_free_rate * discounted_strike)
                    rho = 0.5 * T_minus_t * discounted_strike
                # in the money calls
                elif discounted_spot > discounted_strike:
                    delta = math.exp(-market_env.dividend_yield * T_minus_t)
                    theta = market_env.dividend_yield * discounted_spot - market_env.risk_free_rate * discounted_strike
                    rho = T_minus_t * discounted_strike
                # out of the money calls
                else:
                    delta = 0.0
                    theta = 0.0
                    rho = 0.0
            elif option.option_type == OptionType.PUT:
                # at the money puts    
                if abs(discounted_spot - discounted_strike) < self.TOLERANCE: 
                    delta = -0.5 * math.exp(-market_env.dividend_yield * T_minus_t)
                    theta = 0.5 * (market_env.risk_free_rate * discounted_strike - market_env.dividend_yield * discounted_spot)
                    rho = -0.5 * T_minus_t * discounted_strike   
                # out of the money puts
                elif discounted_spot > discounted_strike:
                    delta = 0.0
                    theta = 0.0
                    rho = 0.0
                # in the money puts    
                else:
                    delta = -math.exp(-market_env.dividend_yield * T_minus_t)
                    theta = market_env.risk_free_rate * discounted_strike - market_env.dividend_yield * discounted_spot
                    rho = -T_minus_t * discounted_strike
            else:
                valid_types = ", ".join(str(t) for t in OptionType)
                raise ValueError(f"Pricing EuropeanOption via BSM is currently implemented only for {valid_types} and you inserted {repr(option.option_type)}!")

            # Gamma and Vega for options with zero vol:    
            if abs(discounted_spot - discounted_strike) < self.TOLERANCE: # at the money options
                gamma = float("inf")
                warnings.warn(f"Beware that for this case (ATM {option.option_type.value} with volatility {market_env.volatility}), Gamma approaches plus infinity!")
                vega = discounted_spot * math.sqrt(T_minus_t/(2*math.pi))
            else: # ITM or OTM options
                gamma = 0.0
                vega = 0.0
       
        return Greeks(delta, gamma, vega, theta, rho)

        