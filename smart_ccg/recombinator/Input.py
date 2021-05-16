from enum import Enum
from Action import Action


class Type(Enum):
    ACTION = "action"
    TABLE = "table"
    CELL = "cell"
    VALUE = "value"
    COLUMN = "column"
    LIST = "list"


class Input:
    def __init__(self, inp_type, content, dsl_format):
        self.type = inp_type
        self.content = content
        self.format = dsl_format

    def generate_dsl_output(self):
        res = self.format
        if type(self.content) is list and self.type[1] == Type.LIST:
            res = res.format(",".join(self.content))
        elif self.type[1] == Type.ACTION and type(self.content[1]) is Action:
            res = res.format(self.content[1].generate_dsl_output(self.content[2]))
        else:
            res.format(self.content)
        return res

    def get_type(self):
        return self.type

    def get_format(self):
        return self.format
