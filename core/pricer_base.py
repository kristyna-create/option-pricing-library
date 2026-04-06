from __future__ import annotations
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod
from dataclasses import replace
from datetime import timedelta
import warnings

if TYPE_CHECKING:
    from core.option_base import BaseOption
from instruments.european import EuropeanOption
from instruments.american import AmericanOption
from instruments.asian import AsianOption
from core.greeks_data import Greeks
from market.environment import MarketEnvironment

class BasePricer(ABC):
    # class-level constants for computing numerical Greeks
    DELTA_BUMP_PCT = 0.01
    VOL_BUMP = 0.001
    RATE_BUMP = 0.001
    THETA_DAYS = 1

    # _calculate_price abstract methods to be implemented by subclasses
    @abstractmethod
    def _calculate_price(self, option: BaseOption, market_env: MarketEnvironment) -> float:
        pass

    # numerical method to be used with numerical pricing engines that will inherit it
    def _calculate_greeks(self, option: BaseOption, market_env: MarketEnvironment) -> Greeks: 
        """Calculates delta, gamma, vega, theta, and rho using finite differences. This is implemented as a concrete method because it is similar for Binomial Trees and Monte Carlo pricers. BlackScholesMertonPricer overrides this method with a closed-form implementation."""    
        if not isinstance(option, (EuropeanOption, AmericanOption, AsianOption)):
            raise TypeError("Greeks computation is currently implemented only for EuropeanOption, AmericanOption, and AsianOption instances!")

        # benchmark price using current market environment
        price = self._calculate_price(option, market_env)

        # delta
        # using central difference
        delta_S = self.DELTA_BUMP_PCT * market_env.spot_price
        market_env_S_plus = replace(market_env, spot_price = market_env.spot_price + delta_S)
        market_env_S_minus = replace(market_env, spot_price = market_env.spot_price - delta_S)
        price_plus = self._calculate_price(option, market_env_S_plus)
        price_minus = self._calculate_price(option, market_env_S_minus)

        delta = (price_plus - price_minus)/(2*delta_S)

        # gamma
        # central difference
        gamma = (price_plus - 2 * price + price_minus)/delta_S**2

        # vega
        market_env_vol_plus = replace(market_env, volatility = market_env.volatility + self.VOL_BUMP)
        price_plus = self._calculate_price(option, market_env_vol_plus)
        # using central difference if (vol - self.VOL_BUMP) is not negative
        if (market_env.volatility - self.VOL_BUMP) >= 0.0:
            market_env_vol_minus = replace(market_env, volatility = market_env.volatility - self.VOL_BUMP)
            price_minus = self._calculate_price(option, market_env_vol_minus)
            vega = (price_plus - price_minus)/(2*self.VOL_BUMP)
        else: # using forward difference
            vega = (price_plus - price)/self.VOL_BUMP

        # theta
        # shift the pricing date forward by self.THETA_DAYS day(s)
        dt = self.THETA_DAYS/365 # annualized theta
        
        if (market_env.pricing_date + timedelta(days=self.THETA_DAYS)) > option.expiry_date: # pricing_date cannot be after expiry_date
            warnings.warn(f"Theta computation: pricing_date = {market_env.pricing_date} + {self.THETA_DAYS} day(s) exceeds the option's expiry_date ({option.expiry_date}). For path-dependent options (e.g. Asian), theta is set to 0.0. For path-independent options, the expiry payoff is used as the forward price.")

            if isinstance(option, (EuropeanOption, AmericanOption)):
                price_tomorrow = option.get_payoff(market_env.spot_price) # get option value at expiry
            elif isinstance(option, AsianOption): 
                price_tomorrow = 0.0 # At maturity, Asian option's value without path history is meaningless, so price_tomorrow = 0.0 and price = 0.0, giving theta = 0.0
            # numerical theta at expiry will be zero by construction 
        else:    
            market_env_tomorrow = replace(market_env, pricing_date = market_env.pricing_date + timedelta(days=self.THETA_DAYS))
            price_tomorrow = self._calculate_price(option, market_env_tomorrow)

        theta = (price_tomorrow - price)/dt

        # rho
        # using central difference
        market_env_r_plus = replace(market_env, risk_free_rate = market_env.risk_free_rate + self.RATE_BUMP)
        market_env_r_minus = replace(market_env, risk_free_rate = market_env.risk_free_rate - self.RATE_BUMP) # we can have negative interest rates
        price_plus = self._calculate_price(option, market_env_r_plus)
        price_minus = self._calculate_price(option, market_env_r_minus)
        
        rho = (price_plus - price_minus)/(2*self.RATE_BUMP)

        return Greeks(float(delta), float(gamma), float(vega), float(theta), float(rho))