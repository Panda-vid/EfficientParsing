from enum import Enum


class TableType(Enum):
    COLUMN = "column"
    TABLE = "table"
    VALUE = "value"
    CONDITION = "condition"
    NO_TABLE_TYPE = "not in table"
