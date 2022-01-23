from __future__ import annotations

from smart_ccg.sem_parser.abstractor.Table import Table
from smart_ccg.sem_parser.abstractor.dependencytree.nodes.LiftableDependencyTreeNode import LiftableDependencyTreeNode


class LiftableValueDependencyTreeNode(LiftableDependencyTreeNode):
    def __init__(self, node_id: int, word: str, word_type: str, parent_id: int, dependency: str, depth: int):
        super().__init__(node_id, word, word_type, parent_id, dependency, depth)

    @classmethod
    def isinstance(cls, node: LiftableDependencyTreeNode):
        return isinstance(node, LiftableValueDependencyTreeNode)

    def lifted(self, table: Table = None) -> str:
        return ""

    def case_lifted(self, table: Table = None) -> str:
        return "<val>"
