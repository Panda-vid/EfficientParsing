from __future__ import annotations
from typing import List

from smart_ccg.sem_parser.abstractor.Table import Table
from smart_ccg.sem_parser.abstractor.dependencytree.nodes.LiftableCaseDependencyTreeNode import \
    LiftableCaseDependencyTreeNode
from smart_ccg.sem_parser.abstractor.dependencytree.nodes.LiftableDependencyTreeNode import LiftableDependencyTreeNode
from smart_ccg.sem_parser.abstractor.dependencytree.nodes.LiftableValueDependencyTreeNode import \
    LiftableValueDependencyTreeNode


class LiftableObjectDependencyTreeNode(LiftableDependencyTreeNode):
    def __init__(self, node_id: int, word: str, word_type: str, parent_id: int, dependency: str, depth: int):
        super().__init__(node_id, word, word_type, parent_id, dependency, depth)

    @classmethod
    def isinstance(cls, node: LiftableDependencyTreeNode):
        return isinstance(node, LiftableObjectDependencyTreeNode)

    def get_object_list(self) -> List[LiftableObjectDependencyTreeNode]:
        res = [self]
        res.extend(self.get_subobjects())
        return res

    def lifted(self, table: Table) -> str:
        if LiftableObjectDependencyTreeNode.isinstance(self.parent):
            res = ""
        elif self.is_successor_of_case():
            res = ""
        else:
            res = self.non_empty_lifted_string(table)
        return res

    def case_lifted(self, table: Table) -> str:
        if LiftableObjectDependencyTreeNode.isinstance(self.parent):
            res = ""
        else:
            res = self.non_empty_lifted_string(table)
        return res

    def non_empty_lifted_string(self, table: Table):
        if table is None:
            res = self.lift_as_column()
        else:
            res = self.resolve_lifted_from(table)
        return res

    def resolve_lifted_from(self, table: Table) -> str:
        if table.has_column(self.word):
            res = self.lift_as_column()
        else:
            res = ""
        return res

    def lift_as_column(self) -> str:
        if self.is_object_enumeration():
            res = "[,column]"
        else:
            res = "[column]"
        return res

    def get_subobjects(self) -> List[LiftableObjectDependencyTreeNode]:
        return [child for child in self.children if isinstance(child, LiftableObjectDependencyTreeNode)]

    def is_object_enumeration(self) -> bool:
        return any([LiftableObjectDependencyTreeNode.isinstance(child) for child in self.children])

    def is_successor_of_case(self) -> bool:
        return self.has_ancestor_with_property(LiftableValueDependencyTreeNode.isinstance)\
               or self.has_ancestor_with_property(LiftableCaseDependencyTreeNode.isinstance)
