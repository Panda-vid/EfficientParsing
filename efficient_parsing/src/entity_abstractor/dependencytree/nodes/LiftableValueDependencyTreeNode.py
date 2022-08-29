from __future__ import annotations

from src.entity_abstractor.dependencytree.nodes.LiftableDependencyTreeNode import LiftableDependencyTreeNode


class LiftableValueDependencyTreeNode(LiftableDependencyTreeNode):
    def __init__(self, node_id: int, word: str, word_type: str, parent_id: int, dependency: str, depth: int):
        super().__init__(node_id, word, word_type, parent_id, dependency, depth)

    @classmethod
    def from_node(cls, node):
        return cls(node.node_id, node.word, node.word_type, node.parent_id, node.dependence, node.depth)

    @classmethod
    def isinstance(cls, node: LiftableDependencyTreeNode) -> bool:
        return isinstance(node, LiftableValueDependencyTreeNode)

    def lifted(self, table=None) -> str:
        return ""

    def case_lifted(self, table=None) -> str:
        return "[value]"
