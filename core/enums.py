from enum import Enum

class OptionType(Enum): # OptionType inherits from Enum class - OptionType is a subclass of Enum
    """Class for defining and checking the option_type attribute of European and American options, defined to avoid misspelling."""
    CALL = "call"
    PUT = "put"

class AsianType(Enum):
    """Class for defining and checking the asian_type attribute of Asian options, defined to avoid misspelling."""
    FIXED_STRIKE = "fixed strike"
    FLOATING_STRIKE = "floating strike"    