from pathlib import Path

import pandas as pd

from typing import List

from src.datamodel.TableType import TableType
from src.entity_abstractor.dependencytree.nodes.LiftableObjectDependencyTreeNode import LiftableObjectDependencyTreeNode
from src.entity_abstractor.utils import normalized_compounds, compounds
from src.util.string_utils import normalize


class Table:
    def __init__(self,
                 data: pd.DataFrame,
                 table_name: str,
                 application_context: str):
        self.data = data
        self.table_name = table_name
        self.columns = [normalize(column_name) for column_name in list(data.columns)]
        self.application_context = application_context

    @classmethod
    def create_from_csv(cls, path: Path, application_context: str):
        dataframe = pd.read_csv(str(path), sep=";")
        return cls(dataframe, path.stem, application_context)

    @classmethod
    def create_test_table_instance(cls, columns: List[str], table_name: str, data: pd.DataFrame):
        table_object = cls(data, table_name, table_name)
        table_object.columns = [normalize(column_name) for column_name in columns]
        return table_object

    def get_table_type(self, node: LiftableObjectDependencyTreeNode):
        if self.is_column(node):
            return TableType.COLUMN
        elif self.is_table(node):
            return TableType.TABLE
        else:
            return TableType.NO_TABLE_TYPE

    def is_column(self, node: LiftableObjectDependencyTreeNode) -> bool:
        return any([self.is_column_name(compound) for compound in normalized_compounds(node)])

    def is_table(self, node: LiftableObjectDependencyTreeNode) -> bool:
        return any([self.is_table_name(compound) for compound in compounds(node)])

    def is_column_name(self, string) -> bool:
        return string in self.columns

    def is_table_name(self, string) -> bool:
        return string == self.table_name

    def get_contexts(self) -> List[str]:
        return [self.table_name, self.application_context]

    def get_entries(self):
        for row in self.data.itertuples():
            yield list(row)[1:]
