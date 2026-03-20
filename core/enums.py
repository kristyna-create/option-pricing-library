from enum import Enum

class OptionType(Enum): # OptionType inherits from Enum class - OptionType is a subclass of Enum
    """Class for defining and checking the option_type attribute of European and American options, defined to avoid misspelling."""
    CALL = "call"
    PUT = "put"