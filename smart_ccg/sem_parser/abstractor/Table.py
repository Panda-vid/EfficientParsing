from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import List

import pandas as pd

from smart_ccg.util.string_utils import normalize
from smart_ccg.util.table_parsing import parse_csv_table


class Table:
    class TableType(Enum):
        COLUMN = "column"
        TABLE = "table"
        NO_TABLE_TYPE = "not in table"

    def __init__(self, columns: List[str], table_name: str, content: pd.DataFrame):
        self.columns = [normalize(column_name) for column_name in columns]
        self.table_name = normalize(table_name)
        self.content = content

    @classmethod
    def create_from_csv(cls, path: Path) -> Table:
        dataframe = parse_csv_table(path)
        return cls.create_from_dataframe(dataframe, path.stem)

    @classmethod
    def create_from_dataframe(cls, dataframe: pd.DataFrame, table_name: str) -> Table:
        columns = list(dataframe)
        return cls(columns, table_name, dataframe)

    def get_lifted_type(self, word: str):
        if self.is_column(word):
            return Table.TableType.COLUMN
        elif self.is_table_name(word):
            return Table.TableType.TABLE
        else:
            return Table.TableType.NO_TABLE_TYPE

    def is_column(self, word: str):
        return normalize(word) in self.columns

    def is_table_name(self, word: str):
        return normalize(word) == self.table_name
