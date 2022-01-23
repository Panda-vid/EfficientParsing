from __future__ import annotations
from typing import List

from smart_ccg.sem_parser.abstractor.Table import Table
from smart_ccg.sem_parser.abstractor.dependencytree.nodes.LiftableDependencyTreeNode import LiftableDependencyTreeNode
from smart_ccg.sem_parser.abstractor.dependencytree.nodes.LiftableValueDependencyTreeNode import \
    LiftableValueDependencyTreeNode


class LiftableCaseDependencyTreeNode(LiftableDependencyTreeNode):
    def __init__(self, node_id: int, word: str, word_type: str, parent_id: int, dependency: str, depth: int):
        super().__init__(node_id, word, word_type, parent_id, dependency, depth)

    @classmethod
    def isinstance(cls, node: LiftableDependencyTreeNode):
        isinstance(node, LiftableCaseDependencyTreeNode)

    def get_case_list(self) -> List[LiftableCaseDependencyTreeNode]:
        res = [self]
        res.extend(self.get_subcases())
        return res

    def lifted(self, table: Table = None) -> str:
        if LiftableCaseDependencyTreeNode.isinstance(self.parent):
            res = ""
        elif self.is_case_enumeration():
            res = "<cases>"
        else:
            res = "<case>"
        return res

    def nodes(self) -> List[LiftableDependencyTreeNode]:
        if LiftableValueDependencyTreeNode.isinstance(self.parent):
            res = self.parent.nodes()
        else:
            res = super(LiftableCaseDependencyTreeNode, self).nodes()
        return res

    def get_subcases(self) -> List[LiftableCaseDependencyTreeNode]:
        return [child for child in self.children if self.isinstance(child)]

    def case_lifted(self, table: Table = None) -> str:
        return super(LiftableCaseDependencyTreeNode, self).lifted(table)

    def get_values(self) -> List[LiftableValueDependencyTreeNode]:
        return [child for child in self.children if LiftableValueDependencyTreeNode.isinstance(child)]

    def is_case_enumeration(self) -> bool:
        return any([isinstance(child, LiftableCaseDependencyTreeNode) for child in self.children])
