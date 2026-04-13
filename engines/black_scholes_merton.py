import numpy as np
from scipy.stats import norm
from dataclasses import replace
import warnings

from core.pricer_base import BasePricer
from core.option_base import BaseOption
from instruments.european import EuropeanOption
from market.environment import MarketEnvironment
from core.enums import OptionType
from core.greeks_data import Greeks

class BlackScholesMertonPricer(BasePricer):

    TOLERANCE = 1e-10 # class-level constant for floating-point comparisons

    def _compute_d1_d2(self, option: EuropeanOption, market_env: MarketEnvironment, T_minus_t: float) -> tuple:
            d1 = (np.log(market_env.spot_price / option.strike_price) + (market_env.risk_free_rate - market_env.dividend_yield + 0.5 * market_env.volatility**2) * T_minus_t) / (market_env.volatility * np.sqrt(T_minus_t))

            d2 = d1 - market_env.volatility * np.sqrt(T_minus_t)

            return (d1, d2)
    
    # The core of this class - pricing method:
    def _calculate_price(self, option: BaseOption, market_env: MarketEnvironment) -> float | np.ndarray:
        """Calculates the option price of EuropeanOption instances according to Black-Scholes-Merton using analytical formulae."""
        if not isinstance(option, EuropeanOption):
            raise TypeError("BSM pricing is mathematically valid only for EuropeanOption instances!")
        
        T_minus_t = option.time_to_maturity(market_env.pricing_date)

        if abs(T_minus_t) < self.TOLERANCE: # Get option value at expiry
            return option.get_payoff(market_env.spot_price)
        else:
            zero_vol_price = np.exp(-market_env.risk_free_rate * T_minus_t) * option.get_payoff(market_env.spot_price * np.exp((market_env.risk_free_rate - market_env.dividend_yield) * T_minus_t))

            safe_vol = np.where(market_env.volatility <= self.TOLERANCE, 0.25, market_env.volatility) # if volatility is zero, dummy volatility of 0.25 will be used for computation not to crash and at the end replaced by zero_vol_price for these inputs

            # Option value for the normal case: some time remains to maturity and volatility is NOT zero
            safe_market_env = replace(market_env, volatility=safe_vol)
            d1, d2 = self._compute_d1_d2(option, safe_market_env, T_minus_t)

            if option.option_type == OptionType.CALL:
                BSM_price = safe_market_env.spot_price * np.exp(-safe_market_env.dividend_yield * T_minus_t) * norm.cdf(d1) - option.strike_price * np.exp(-safe_market_env.risk_free_rate * T_minus_t) * norm.cdf(d2)
            elif option.option_type == OptionType.PUT:
                BSM_price = -safe_market_env.spot_price * np.exp(-safe_market_env.dividend_yield * T_minus_t) * norm.cdf(-d1) + option.strike_price * np.exp(-safe_market_env.risk_free_rate * T_minus_t) * norm.cdf(-d2)
            else:
                valid_types = ", ".join(str(t) for t in OptionType)
                raise ValueError(f"Pricing EuropeanOption via BSM is currently implemented only for {valid_types} and you inserted {repr(option.option_type)}!")
            
            return np.where(market_env.volatility > self.TOLERANCE, BSM_price, zero_vol_price)

    # Closed-form Greeks for BSM - overrides _calculate_greeks() method from the parent (BasePricer) 
    def _calculate_greeks(self, option: BaseOption, market_env: MarketEnvironment) -> Greeks:
        """Calculates delta, gamma, vega, theta, rho of EuropeanOption instances according to Black-Scholes-Merton analytical formulas."""
        if not isinstance(option, EuropeanOption):
            raise TypeError("BSM pricing is mathematically valid only for EuropeanOption instances!")
        
        # Solve for scalar/array broadcasting mismatches that might occur
        # Create a dummy template that adopts the maximum broadcasted shape of all inputs
        shape_template = 0.0 * (market_env.spot_price + market_env.volatility + market_env.risk_free_rate + market_env.dividend_yield)
        
        T_minus_t = option.time_to_maturity(market_env.pricing_date)

        if abs(T_minus_t) < self.TOLERANCE: # Greeks at expiry
            is_atm_spot = np.abs(market_env.spot_price - option.strike_price) < self.TOLERANCE

            if option.option_type == OptionType.CALL:
                delta = np.where(is_atm_spot, # ATM calls
                                 0.5, # Delta for ATM calls
                                 np.where(market_env.spot_price > option.strike_price, 1.0, 0.0) # Delta for ITM calls and else for OTM calls
                                 )
                theta = np.where(is_atm_spot, # ATM calls
                                 -np.inf, # Theta for ATM calls
                                 np.where(market_env.spot_price > option.strike_price, # ITM calls
                                          market_env.dividend_yield * market_env.spot_price - market_env.risk_free_rate * option.strike_price, # Theta for ITM calls
                                          0.0) # and else for OTM calls
                                 ) 
            elif option.option_type == OptionType.PUT:
                delta = np.where(is_atm_spot, # ATM puts
                                 -0.5, # Delta for ATM puts
                                 np.where(market_env.spot_price > option.strike_price, 0.0, -1.0) # Delta for OTM puts and else for ITM puts 
                                 )
                theta = np.where(is_atm_spot, # ATM puts
                                 -np.inf, # Theta for ATM puts
                                 np.where(market_env.spot_price > option.strike_price, # OTM puts
                                          0.0, # Theta for OTM puts
                                          market_env.risk_free_rate * option.strike_price - market_env.dividend_yield * market_env.spot_price) # Theta for ITM puts
                                 )
            else:
                valid_types = ", ".join(str(t) for t in OptionType)
                raise ValueError(f"Pricing EuropeanOption via BSM is currently implemented only for {valid_types} and you inserted {repr(option.option_type)}!") 

            # Gamma for options at expiry (calls and puts):
            gamma = np.where(is_atm_spot, # ATM options
                             np.inf, # Gamma for ATM options at expiry
                             0.0)   # Gamma for ITM or OTM options at expiry              

            # Vega and Rho for options at expiry:
            vega = shape_template # array of zeros defined above
            rho = shape_template # array of zeros defined above

            # Issue a warning if any inputs are ATM
            if np.any(is_atm_spot):
                if isinstance(shape_template, np.ndarray):
                    msg = (f"One or more inputs result in an At-The-Money (ATM) spot price at expiry. "
                           f"In this limit, Gamma approaches plus infinity and Theta approaches minus infinity.")
                else:
                    msg = (f"The input results in an At-The-Money (ATM) spot price at expiry. "
                           f"In this limit, Gamma approaches plus infinity and Theta approaches minus infinity.")
                warnings.warn(msg)

        else: # Some time remains to maturity, have to handle the edge case of zero volatility
            #-------------------------- Greeks for the zero-vol edge case
            # Zero-Vol Greeks
            # Theoretical case Greeks: when volatility is zero
            discounted_spot = market_env.spot_price * np.exp(-market_env.dividend_yield * T_minus_t)
            discounted_strike = option.strike_price * np.exp(-market_env.risk_free_rate * T_minus_t)

            # Forward price equals strike (ATM Forward)
            is_atm_forward = np.abs(discounted_spot - discounted_strike) < self.TOLERANCE
            # Volatility is effectively zero
            is_zero_vol = market_env.volatility <= self.TOLERANCE

            if option.option_type == OptionType.CALL:
                delta_zero = np.where(is_atm_forward, # ATM calls
                                      0.5 * np.exp(-market_env.dividend_yield * T_minus_t), # Delta for ATM calls
                                      np.where(discounted_spot > discounted_strike, # ITM calls
                                                np.exp(-market_env.dividend_yield * T_minus_t), # Delta for ITM calls
                                                0.0) # Delta for OTM calls
                                 )
                theta_zero = np.where(is_atm_forward, # ATM calls
                                      0.5 * (market_env.dividend_yield * discounted_spot - market_env.risk_free_rate * discounted_strike), # Theta for ATM calls
                                      np.where(discounted_spot > discounted_strike, # ITM calls
                                                market_env.dividend_yield * discounted_spot - market_env.risk_free_rate * discounted_strike, # Theta for ITM calls
                                                0.0) # Theta for OTM calls
                                 )
                rho_zero = np.where(is_atm_forward, # ATM calls
                                      0.5 * T_minus_t * discounted_strike, # Rho for ATM calls
                                      np.where(discounted_spot > discounted_strike, # ITM calls
                                                T_minus_t * discounted_strike, # Rho for ITM calls
                                                0.0) # Rho for OTM calls
                                 )
            elif option.option_type == OptionType.PUT:
                delta_zero = np.where(is_atm_forward, # ATM puts
                                      -0.5 * np.exp(-market_env.dividend_yield * T_minus_t), # Delta for ATM puts
                                      np.where(discounted_spot > discounted_strike, # OTM puts
                                                0.0, # Delta for OTM puts
                                                -np.exp(-market_env.dividend_yield * T_minus_t)) # Delta for ITM puts
                                 )
                theta_zero = np.where(is_atm_forward, # ATM puts
                                      0.5 * (market_env.risk_free_rate * discounted_strike - market_env.dividend_yield * discounted_spot), # Theta for ATM puts
                                      np.where(discounted_spot > discounted_strike, # OTM puts
                                                0.0, # Theta for OTM puts
                                                market_env.risk_free_rate * discounted_strike - market_env.dividend_yield * discounted_spot) # Theta for ITM puts
                                 )
                rho_zero = np.where(is_atm_forward, # ATM puts
                                      -0.5 * T_minus_t * discounted_strike, # Rho for ATM puts
                                      np.where(discounted_spot > discounted_strike, # OTM puts
                                                0.0, # Rho for OTM puts
                                                -T_minus_t * discounted_strike) # Rho for ITM puts
                                 )
            else:
                valid_types = ", ".join(str(t) for t in OptionType)
                raise ValueError(f"Pricing EuropeanOption via BSM is currently implemented only for {valid_types} and you inserted {repr(option.option_type)}!")
            
            # Gamma and Vega for options with zero vol (calls and puts):
            gamma_zero = np.where(is_atm_forward, # ATM options
                                  np.inf, # Gamma for ATM Forward options when volatility is zero
                                  0.0) # ITM or OTM options
            
            vega_zero = np.where(is_atm_forward, # ATM options
                                  discounted_spot * np.sqrt(T_minus_t/(2*np.pi)), # Vega for ATM Forward options when volatility is zero
                                  0.0) # ITM or OTM options

            # Warning when Gamma approaches infinity
            if np.any(is_atm_forward & is_zero_vol):
                if isinstance(shape_template, np.ndarray):
                    msg = (f"One or more inputs result in an At-The-Money (ATM) forward price under zero volatility. "
                           f"In this deterministic limit, Gamma approaches plus infinity.")
                else:
                    msg = (f"The input results in an At-The-Money (ATM) forward price under zero volatility. "
                           f"In this deterministic limit, Gamma approaches plus infinity.")
                warnings.warn(msg)           
            
            #-------------------------- Normal case Greeks
            # Normal case Greeks: some time remains to maturity and volatility is not zero
            safe_vol = np.where(is_zero_vol, 0.25, market_env.volatility) # if volatility is zero, dummy volatility of 0.25 will be used for computation not to crash and at the end replaced by Zero-Vol Greeks for these inputs

            # BSM Greeks for the normal case: some time remains to maturity and volatility is NOT zero
            safe_market_env = replace(market_env, volatility=safe_vol)
            d1, d2 = self._compute_d1_d2(option, safe_market_env, T_minus_t)

            if option.option_type == OptionType.CALL:
                delta_bsm = np.exp(-safe_market_env.dividend_yield * T_minus_t) * norm.cdf(d1)
                
                theta_bsm = -safe_market_env.volatility * safe_market_env.spot_price * np.exp(-safe_market_env.dividend_yield * T_minus_t) * norm.pdf(d1) / (2 * np.sqrt(T_minus_t)) + safe_market_env.dividend_yield * safe_market_env.spot_price * norm.cdf(d1) * np.exp(-safe_market_env.dividend_yield * T_minus_t) - safe_market_env.risk_free_rate * option.strike_price * np.exp(-safe_market_env.risk_free_rate * T_minus_t) * norm.cdf(d2)  
                
                rho_bsm = option.strike_price * T_minus_t * np.exp(-safe_market_env.risk_free_rate * T_minus_t) * norm.cdf(d2)
            elif option.option_type == OptionType.PUT:
                delta_bsm = np.exp(-safe_market_env.dividend_yield * T_minus_t) * (norm.cdf(d1) - 1)
                
                theta_bsm = -safe_market_env.volatility * safe_market_env.spot_price * np.exp(-safe_market_env.dividend_yield * T_minus_t) * norm.pdf(-d1) / (2 * np.sqrt(T_minus_t)) - safe_market_env.dividend_yield * safe_market_env.spot_price * norm.cdf(-d1) * np.exp(-safe_market_env.dividend_yield * T_minus_t) + safe_market_env.risk_free_rate * option.strike_price * np.exp(-safe_market_env.risk_free_rate * T_minus_t) * norm.cdf(-d2) 

                rho_bsm = -option.strike_price * T_minus_t * np.exp(-safe_market_env.risk_free_rate * T_minus_t) * norm.cdf(-d2)
            else:
                valid_types = ", ".join(str(t) for t in OptionType)
                raise ValueError(f"Pricing EuropeanOption via BSM is currently implemented only for {valid_types} and you inserted {repr(option.option_type)}!")

            # Gamma and Vega are identical for European calls and puts
            gamma_bsm = np.exp(-safe_market_env.dividend_yield * T_minus_t) * norm.pdf(d1) / (safe_market_env.volatility * safe_market_env.spot_price * np.sqrt(T_minus_t))   
            vega_bsm = safe_market_env.spot_price * np.sqrt(T_minus_t) * np.exp(-safe_market_env.dividend_yield * T_minus_t) * norm.pdf(d1)

            #-------------------------- Merge zero-vol edge case and normal case Greeks
            delta = np.where(market_env.volatility > self.TOLERANCE, delta_bsm, delta_zero)
            gamma = np.where(market_env.volatility > self.TOLERANCE, gamma_bsm, gamma_zero)
            vega = np.where(market_env.volatility > self.TOLERANCE, vega_bsm, vega_zero)
            theta = np.where(market_env.volatility > self.TOLERANCE, theta_bsm, theta_zero)
            rho = np.where(market_env.volatility > self.TOLERANCE, rho_bsm, rho_zero)

        
        # If spot was a scalar but vol was an array and option at expiry, delta, gamma, theta are currently scalars
        # Adding shape_template forces numpy to broadcast that scalar into an array of the correct size   
        # the shape_template is just an array of 0.0 of the correct shape
        return Greeks(
        delta = delta + shape_template,
        gamma = gamma + shape_template,
        vega  = vega  + shape_template,
        theta = theta + shape_template,
        rho   = rho   + shape_template
    )

        