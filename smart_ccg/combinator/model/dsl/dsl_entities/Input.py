from typing import Protocol


class Input(Protocol):
    def type(self):
        """An Input must have a type associated with it."""

    def dsl_output(self) -> str:
        """
        An Input must be able to generate its output for the dsl.
        The Input must not be able to know if it is surrounded by specific brackets
        only how one input instance looks like in the DSL.
        e.g. An Assignment must be able to give an output looking like this:
             asssignee=value
        """
