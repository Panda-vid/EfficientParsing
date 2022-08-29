from typing import List

from smart_ccg.combinator.model.dsl.dsl_entities.Input import Input
from smart_ccg.combinator.model.dsl.dsl_entities.Type import Type


class ProtoAction:
    def __init__(self, dsl_text: str, input_types: List[Type], input_formats: List[str]):
        self.dsl_text = dsl_text
        self.input_types = input_types
        self.input_formats = input_formats

    def is_valid(self, inputs: List[Input]):
        return len(inputs) == len(self.input_types) and \
               len(inputs) == len(self.input_formats) and \
               all(inp.type() == action_input_type for inp, action_input_type in zip(inputs, self.input_types))

    def type(self):
        return Type.ACTION, self.dsl_text
