from smart_ccg.combinator.model.dsl.dsl_entities.Type import Type


class SimpleInput:
    def __init__(self, input_type: Type, content):
        self.input_type = input_type
        self.content = content

    def dsl_output(self) -> str:
        return self.content

    def type(self):
        return self.input_type
