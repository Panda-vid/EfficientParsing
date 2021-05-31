from typing import List

from smart_ccg.combinator.model.dsl.dsl_entities.Input import Input
from smart_ccg.combinator.model.dsl.dsl_entities.Type import Type


class DSLList:
    def __init__(self, inputs: List[Input]):
        self.contents = inputs

    def dsl_output(self) -> str:
        output = []
        for member in self.contents:
            output.append(member.dsl_output())
        return ",".join(output)

    def type(self):
        return Type.LIST, self.contents[0].type()