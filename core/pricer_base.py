from __future__ import annotations
from typing import TYPE_CHECKING

from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from core.option_base import BaseOption
from core.greeks_data import Greeks
from market.environment import MarketEnvironment

class BasePricer(ABC):
    # two abstract methods to be implemented by subclasses

    @abstractmethod
    def _calculate_price(self, option: BaseOption, market_env: MarketEnvironment) -> float:
        pass

    @abstractmethod
    def _calculate_greeks(self, option: BaseOption, market_env: MarketEnvironment) -> Greeks:
        pass