from smart_ccg.combinator.model.dsl.DSL import DSL
from smart_ccg.combinator.model.dsl.dsl_entities.SimpleInput import SimpleInput
from smart_ccg.combinator.model.dsl.dsl_entities.Type import Type


class Condition:
    def __init__(self, dsl: DSL, conditional: str, selected_column: SimpleInput, condition_value: SimpleInput):
        self.dsl_text = dsl.condition
        self.conditional = conditional
        self.selected_column = selected_column
        self.condition_value = condition_value

    def dsl_output(self):
        return "{}{}{}{}".format(self.dsl_text, self.selected_column.dsl_output(), self.conditional,
                                 self.condition_value.dsl_output())

    def type(self):
        return Type.CONDITION, self.condition_value.type()
