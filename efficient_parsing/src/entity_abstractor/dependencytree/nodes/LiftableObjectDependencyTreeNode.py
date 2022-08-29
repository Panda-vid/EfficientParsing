from __future__ import annotations
from typing import List

from src.datamodel.Table import TableType
from src.entity_abstractor.dependencytree.nodes.LiftableDependencyTreeNode import LiftableDependencyTreeNode
from src.entity_abstractor.dependencytree.nodes.LiftableValueDependencyTreeNode import LiftableValueDependencyTreeNode
from src.util.string_utils import normalize


class LiftableObjectDependencyTreeNode(LiftableDependencyTreeNode):
    def __init__(self, node_id: int, word: str, word_type: str, parent, dependency: str, depth: int):
        super().__init__(node_id, word, word_type, parent, dependency, depth)

    @classmethod
    def isinstance(cls, node: LiftableDependencyTreeNode) -> bool:
        return isinstance(node, LiftableObjectDependencyTreeNode)

    def get_object_list(self) -> List[LiftableObjectDependencyTreeNode]:
        return [node for node in self.nodes() if isinstance(node, LiftableObjectDependencyTreeNode)]

    def lifted(self, table=None) -> str:
        if LiftableObjectDependencyTreeNode.isinstance(self.parent) \
                and self.get_table_type(table) == self.parent.get_table_type(table):
            res = ""
        elif self.is_successor_of_case() or self.is_oblique_predecessor_of_case():
            res = ""
        else:
            res = self.non_empty_lifted_string(table)
        return res

    def case_lifted(self, table=None) -> str:
        if LiftableObjectDependencyTreeNode.isinstance(self.parent):
            res = ""
        else:
            res = self.non_empty_lifted_string(table)
        return res

    def non_empty_lifted_string(self, table):
        return self.resolve_lifted_from(table)

    def resolve_lifted_from(self, table) -> str:
        table_type = self.get_table_type(table)
        if table_type == TableType.COLUMN:
            res = self.lift_as_column(table)
        elif table_type == TableType.TABLE:
            res = self.lift_as_table(table)
        else:
            res = self.word
        return res

    def get_table_type_from_compound(self):
        for node in self.nodes():
            if normalize(node.word) == "table":
                return TableType.TABLE
            if normalize(node.word) == "column":
                return TableType.COLUMN
        if LiftableObjectDependencyTreeNode.isinstance(self.parent):
            return self.parent.get_table_type_from_compound()

    def lift_as_column(self, table) -> str:
        if self.is_object_enumeration(table):
            res = "[,column]"
        else:
            res = "[column]"
        return res

    def lift_as_table(self, table) -> str:
        if self.is_object_enumeration(table):
            res = self.word
        else:
            res = "[table]"
        return res

    def get_table_type(self, table) -> TableType:
        if table is not None:
            return table.get_table_type(self)
        else:
            return self.get_table_type_from_compound()

    def is_object_enumeration(self, table) -> bool:
        return any([
            LiftableObjectDependencyTreeNode.isinstance(child) and
            self.get_table_type(table) == child.get_table_type(table)
            for child in self.children
        ])

    def is_successor_of_case(self) -> bool:
        return self.has_ancestor_with_property(LiftableValueDependencyTreeNode.isinstance)\
               or self.has_ancestor_with_property(LiftableDependencyTreeNode.is_valid_condition)

    def contains_case(self) -> bool:
        return any(LiftableValueDependencyTreeNode.isinstance(child) for child in self.children)

    def is_oblique_predecessor_of_case(self) -> bool:
        return self.is_oblique() and any([child.is_valid_condition() for child in self.children])
