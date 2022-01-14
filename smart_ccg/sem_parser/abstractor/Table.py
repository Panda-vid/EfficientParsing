from __future__ import annotations
from pathlib import Path
from typing import List

import pandas as pd
import inflection

from smart_ccg.util.table_parsing import parse_csv_table


class Table:
    def __init__(self, columns: List[str], table_name: str, content: pd.DataFrame):
        self.columns = columns
        self.table_name = table_name
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


