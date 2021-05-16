from Input import Input, Type


class Action:
    def __init__(self, dsl_text, input_types, input_formats):
        self.type_list = input_types
        self.action_dsl_text = dsl_text
        self.input_formats = input_formats

    def transform_to_input(self, inputs):
        action_format = self.action_dsl_text + " " + " ".join(self.input_formats)
        Input((Type.ACTION, self.action_dsl_text), (self, inputs), action_format)

    def generate_dsl_output(self, inputs):
        output = [self.action_dsl_text]
        if self.check_correctness(inputs):
            for inp in inputs:
                output.append(inp.generate_dsl_output())
        return " ".join(output)

    def check_correctness(self, inputs):
        res = True
        for i in range(len(inputs)):
            if inputs[i].get_type() != self.type_list[i] \
                    or inputs[i].get_format() != self.input_formats[i]:
                res = False
        return res
