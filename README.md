# Option Pricing Library

A Python library for pricing European options and computing Greeks using Black-Scholes-Merton (BSM), Binomial Trees (CRR model), and Monte Carlo simulation. Built from scratch with clean OOP architecture — no AI-generated code.

## Features
- **Separation of instruments and engines** — options define contract logic (payoffs), pricers implement the math. Swap pricing models at runtime without changing instrument code.
- **Risk management - Greeks** — closed-form delta, gamma, vega, theta, and rho with correct limiting behavior at expiry and zero volatility from BSM, and numerical Greeks.
- **Robust edge case handling** — maturity limits, zero-volatility pricing, input validation with clear error messages.
- **Put-call parity validated** — tested against known reference values and arbitrage-free relationships using `pytest`.
- **Extensible for exotics** — architecture designed so adding new option types (barriers, Asians) requires only new subclasses, no changes to existing code.

## Architecture
Designed using UML class diagrams before writing any code. See below for the full diagram.

## Architecture
```mermaid
classDiagram
    class BaseOption {
        <<abstract>>
        +float strike_price
        +date expiry_date
        +price(pricer: BasePricer, market_env: MarketEnvironment) float
        +greeks(pricer: BasePricer, market_env: MarketEnvironment) Greeks
        +get_payoff(spot_price: float)* float
        +time_to_maturity(pricing_date: date) float

    }

class OptionType {
    <<enumeration>>
    CALL 
    PUT
}    

class EuropeanOption {
    +OptionType option_type
    +get_payoff(spot_price: float) float
}

BaseOption <|-- EuropeanOption
EuropeanOption --> OptionType

class MarketEnvironment {
    +float spot_price
    +float risk_free_rate
    +float volatility
    +float dividend_yield
    +date pricing_date
}

class Greeks {
    +float delta
    +float gamma
    +float vega
    +float theta
    +float rho
}

class BasePricer {
    <<abstract>>
    -_calculate_price(option: BaseOption, market_env: MarketEnvironment)* float
    -_calculate_greeks(option: BaseOption, market_env: MarketEnvironment)* Greeks
}

class BlackScholesMertonPricer {
    -_calculate_price(option: BaseOption, market_env: MarketEnvironment) float
    -_calculate_greeks(option: BaseOption, market_env: MarketEnvironment) Greeks    
}

BasePricer <|-- BlackScholesMertonPricer

class MonteCarloPricer {
    +int num_paths
    +int num_steps
    +int random_seed
    -_calculate_price(option: BaseOption, market_env: MarketEnvironment) float
    -_calculate_greeks(option: BaseOption, market_env: MarketEnvironment) Greeks
}

BasePricer <|-- MonteCarloPricer

class BinomialTreePricer {
    +int num_tree_steps
    -_calculate_price(option: BaseOption, market_env: MarketEnvironment) float
    -_calculate_greeks(option: BaseOption, market_env: MarketEnvironment) Greeks
}

BasePricer <|-- BinomialTreePricer

BaseOption ..> BasePricer : uses
BaseOption ..> MarketEnvironment : uses 
BaseOption ..> Greeks : returns 

BasePricer ..> BaseOption : uses 
BasePricer ..> MarketEnvironment : uses 
BasePricer ..> Greeks : returns
```
