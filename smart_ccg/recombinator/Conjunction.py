from Input import Input


def check_correctness(left_side, right_side):
    return left_side.get_type() == right_side.get_type()


class Conjunction:
    def __init__(self, conjunction_dsl_text):
        self.conjunction_dsl_text = conjunction_dsl_text

    def generate_dsl_output(self, left_input, right_input):
        if check_correctness(left_input, right_input):
            return left_input.generate_dsl_output() + self.conjunction_dsl_text + right_input.generate_dsl_output()
