from typing import List

from smart_ccg.combinator.model.dsl.dsl_entities.Input import Input
from smart_ccg.combinator.model.dsl.dsl_entities.ProtoAction import ProtoAction
from smart_ccg.combinator.model.dsl.dsl_entities.Type import Type


class Action:
    def __init__(self, proto_action: ProtoAction, inputs: List[Input]):
        self.dsl_text = proto_action.dsl_text
        self.input_formats = proto_action.input_formats
        self.inputs = inputs

    def dsl_output(self):
        output = [self.dsl_text]
        for i in range(len(self.inputs)):
            inp, inp_format = self.inputs[i], self.input_formats[i]
            output.append(inp_format.format(inp.dsl_output()))
        return " ".join(output)

    def type(self):
        return Type.ACTION, self.dsl_text
