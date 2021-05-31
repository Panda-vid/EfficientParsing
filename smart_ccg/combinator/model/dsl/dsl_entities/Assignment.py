from smart_ccg.combinator.model.dsl.dsl_entities.Input import Input
from smart_ccg.combinator.model.dsl.dsl_entities.Type import Type


class Assignment:
    def __init__(self, assignee: Input, value: Input):
        self.assignee = assignee
        self.value = value

    def dsl_output(self) -> str:
        return "{}={}".format(self.assignee.dsl_output(), self.value.dsl_output())

    def type(self):
        return Type.ASSIGNMENT, (self.assignee.type(), self.value.type())
