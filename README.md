# Option Pricing Library

This library will be a professional Python implementation of various option pricing models for different types of options. 
It is written using **Object-Oriented Programming (OOP)**. It is my personal project on which I am working because I am interested in option pricing methods and I want to get better in object-oriented programming in Python.

## Key Features
- **Contract-Based Design**: Instruments store only contract logic (payoffs).
- **Model Independence**: Swap between Black-Scholes, Monte Carlo, and Binomial Trees at runtime.
- **Risk Management**: Dedicated `Greeks` container for managing first and second-order sensitivities.

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
