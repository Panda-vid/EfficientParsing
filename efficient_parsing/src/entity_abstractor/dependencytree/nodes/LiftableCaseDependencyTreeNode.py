from __future__ import annotations
from typing import List

from src.entity_abstractor.dependencytree.nodes.LiftableDependencyTreeNode import LiftableDependencyTreeNode
from src.entity_abstractor.dependencytree.nodes.LiftableValueDependencyTreeNode import LiftableValueDependencyTreeNode


class LiftableCaseDependencyTreeNode(LiftableDependencyTreeNode):
    """
    This class represents nodes in a dependency tree which are associated with sentence cases.
    """
    def __init__(self, node_id: int, word: str, word_type: str, parent_id: int, dependency: str, depth: int):
        super().__init__(node_id, word, word_type, parent_id, dependency, depth)

    @classmethod
    def isinstance(cls, node: LiftableDependencyTreeNode) -> bool:
        """
        Check whether input node is a case node.
        :param node:
        :return:
        """
        return isinstance(node, LiftableCaseDependencyTreeNode)

    def lifted(self, table=None) -> str:
        """
        Get the representation of the word corresponding to this node in the lifted string.
        :param table: The active table in the parser's context.
        :return:
        """
        if LiftableCaseDependencyTreeNode.isinstance(self.parent) or not self.is_valid_condition():
            res = ""
        elif self.is_case_enumeration():
            res = "[,condition]"
        else:
            res = "[condition]"
        return res

    def nodes(self) -> List[LiftableDependencyTreeNode]:
        """
        Get the nodes corresponding to this case dependency.
        :return:
        """
        if LiftableValueDependencyTreeNode.isinstance(self.parent) or self.parent.is_oblique():
            res = self.parent.nodes()
        else:
            res = super(LiftableCaseDependencyTreeNode, self).nodes()
        return res

    def get_lifted_value_node_from_children(self) -> LiftableValueDependencyTreeNode:
        """
        Retrieve the value node associated with the condition defined by this sentence case.
        :return:
        """
        return [child for child in self.children if LiftableValueDependencyTreeNode.isinstance(child)][0] \
            if len(self.children) > 0 else None

    def get_subcases(self) -> List[LiftableCaseDependencyTreeNode]:
        """
        Get all subcases of this sentence case (i.e. if there are multiple conditions separated by 'and' or 'or').
        :return:
        """
        return [child for child in self.children if self.isinstance(child)]

    def case_lifted(self, table=None) -> str:
        """
        Get the representation of the word corresponding to this node in the lifted condition string.
        :param table: The active table in the parser's context.
        :return:
        """
        return super(LiftableCaseDependencyTreeNode, self).lifted(table)

    def is_valid_condition(self) -> bool:
        """
        Check whether this sentence case is a condition where an object is related to some value.
        :return:
        """
        # We do not need to check for a surrounding object since this is a prerequisite for getting assigned this class.
        return self.has_surrounding_value()

    def is_case_enumeration(self) -> bool:
        """
        Check whether this case contains multiple subcases.
        :return:
        """
        return any([LiftableCaseDependencyTreeNode.isinstance(child) for child in self.children])

    def has_surrounding_value(self) -> bool:
        """
        Check whether there is a value in the neighboring nodes, the parent node and all child nodes.
        :return:
        """
        return self.has_neighboring_value() or self.has_parent_value() or self.has_child_value()

    def has_neighboring_value(self) -> bool:
        """
        Check whether there is a value node neighboring this node.
        :return:
        """
        return all(LiftableValueDependencyTreeNode.isinstance(neighbor) for neighbor in self.get_neighbors()) \
            and len(self.get_neighbors()) > 0

    def has_child_value(self) -> bool:
        """
        Check whether a child is a value node.
        :return:
        """
        return all(LiftableValueDependencyTreeNode.isinstance(child) for child in self.children) \
            and len(self.children) > 0

    def has_parent_value(self) -> bool:
        """
        Check whether the parent is a value node.
        :return:
        """
        return LiftableValueDependencyTreeNode.isinstance(self.parent)
