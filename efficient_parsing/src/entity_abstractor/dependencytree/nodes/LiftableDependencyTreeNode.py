from __future__ import annotations

import copy
from typing import List, Callable, Any


class LiftableDependencyTreeNode:
    """
    This parent class models any dependency tree node inside our liftable dependency tree.
    """
    def __init__(self, node_id: int,
                 word: str,
                 word_type: str,
                 parent_id: int,
                 dependency: str,
                 depth=0,
                 children=None):
        """

        :param node_id:
        :param word:
        :param word_type: The word type e.g. VB verb etc..
        :param parent_id: The parent node id.
        :param dependency: The universal dependency associated with this word.
        :param depth:
        :param children:
        """
        if children is None:
            children = []

        self.node_id = node_id
        self.dependency = dependency
        self.word = word
        self.word_type = word_type
        self.children = children
        self.depth = depth
        self.parent: Any = parent_id
        self.lifted_string = word
        self.case_lifted_string = ""

    @staticmethod
    def isinstance(node: Any) -> bool:
        """
        Check whether input node is a dependency tree node.
        :param node:
        :return:
        """
        return isinstance(node, LiftableDependencyTreeNode)

    def case_lifted(self, table=None) -> str:
        """
        Get the representation of the word corresponding to this node in the lifted condition string.
        :param table: The active table in the parser's context.
        :return:
        """
        return self.lifted(table)

    def lifted(self, table=None) -> str:
        """
        Get the representation of the word corresponding to this node in the lifted string.
        :param table: The active table in the parser's context.
        :return:
        """
        if self.dependency == "case":
            return self.word if self.parent.contains_case() else ""
        return self.word

    def nodes(self) -> List[LiftableDependencyTreeNode]:
        """
        All descendents of this node and the node itself.
        :return:
        """
        res = [self]
        for child in self.children:
            if child.is_leaf():
                res.append(child)
            else:
                res += child.nodes()
        res.sort()
        return res

    def add_child(self, child: LiftableDependencyTreeNode) -> None:
        self.children.append(child)
        child.set_parent(self)

    def set_parent(self, parent: LiftableDependencyTreeNode) -> None:
        self.parent = parent

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def is_sentence_root(self) -> bool:
        """
        Check whether this node is the root of a liftable dependency tree.
        :return:
        """
        return self.parent == 0 and self.dependency == "ROOT"

    def is_oblique(self) -> bool:
        return self.dependency == "obl"

    def has_parent(self) -> bool:
        return isinstance(self.parent, LiftableDependencyTreeNode)

    def has_ancestor_with_property(self, property_function: Callable[[LiftableDependencyTreeNode], bool]) -> bool:
        """
        Check whether an ancestor has a property defined by the property function given to this method.
        :param property_function:
        :return:
        """
        return any([property_function(ancestor) for ancestor in self.get_ancestors()])

    def get_ancestors(self) -> List[LiftableDependencyTreeNode]:
        res = []
        if self.has_parent():
            res.append(self.parent)
            res.extend(self.parent.get_ancestors())
        return res

    def get_neighbors(self) -> List[LiftableDependencyTreeNode]:
        res = copy.deepcopy(self.parent.children)
        res.remove(self)
        return res

    def get_position(self) -> int:
        return self.node_id

    def get_word(self) -> str:
        return self.word

    def is_valid_condition(self) -> bool:
        """
        Check whether this node is a condition where an object is related to some value.
        :return:
        """
        return False

    def contains_case(self) -> bool:
        """
        Check whether this node contains a sentence case.
        :return:
        """
        return False

    def __eq__(self, other) -> bool:
        res = False
        if LiftableDependencyTreeNode.isinstance(other):
            node_id_equals = self.node_id == other.node_id
            dependency_equals = self.dependency == other.dependency
            word_equals = self.word == other.word
            depth_equals = self.depth == other.depth
            parent_equals = self.parent == other.parent
            res = node_id_equals and dependency_equals and word_equals and depth_equals and parent_equals
        return res

    def __sub__(self, other: LiftableDependencyTreeNode):
        if other in self.children:
            self.children.remove(other)
        else:
            self.children = [child - other for child in self.children]
        return self

    def __gt__(self, other) -> bool:
        return self.node_id > other.node_id

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}[{self.node_id}/{self.word_type}/{self.dependency}/{self.lifted(None)}/" + \
               f"{self.get_word()}]"
