from __future__ import annotations

from src.entity_abstractor.dependencytree.nodes.LiftableDependencyTreeNode import LiftableDependencyTreeNode


class LiftableValueDependencyTreeNode(LiftableDependencyTreeNode):
    """
    This class represents nodes in a dependency tree which are associated with sentence values.
    """
    def __init__(self, node_id: int, word: str, word_type: str, parent_id: int, dependency: str, depth: int):
        super().__init__(node_id, word, word_type, parent_id, dependency, depth)

    @classmethod
    def from_node(cls, node):
        """
        Create a value node from any liftable dependency tree node.
        :param node:
        :return:
        """
        return cls(node.node_id, node.word, node.word_type, node.parent_id, node.dependence, node.depth)

    @classmethod
    def isinstance(cls, node: LiftableDependencyTreeNode) -> bool:
        """
        Check whether input node is a value node.
        :param node:
        :return:
        """
        return isinstance(node, LiftableValueDependencyTreeNode)

    def lifted(self, table=None) -> str:
        """
        Get the representation of the word corresponding to this node in the lifted string.
        :param table: The active table in the parser's context.
        :return:
        """
        return ""

    def case_lifted(self, table=None) -> str:
        """
        Get the representation of the word corresponding to this node in the lifted condition string.
        :param table: The active table in the parser's context.
        :return:
        """
        return "[value]"
