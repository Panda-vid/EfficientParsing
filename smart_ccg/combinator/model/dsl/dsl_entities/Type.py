from enum import Enum


class Type(Enum):
    TABLE = "table"
    VALUE = "value"
    COLUMN = "column"
    CONDITION = "condition"
    ASSIGNMENT = "="
    ACTION = "action"
    LIST = "list"
