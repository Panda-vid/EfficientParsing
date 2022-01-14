from __future__ import annotations

from typing import List, Callable, Any

from smart_ccg.sem_parser.abstractor.Table import Table


class LiftableDependencyTreeNode:

    def __init__(self, node_id: int,
                 word: str,
                 word_type: str,
                 parent_id: int,
                 dependency: str,
                 depth=0,
                 children=None):
        if children is None:
            children = []

        self.node_id = node_id
        self.dependency = dependency
        self.word = word
        self.word_type = word_type
        self.children = children
        self.depth = depth
        self.parent: Any = parent_id

    @classmethod
    def isinstance(cls, node: Any) -> bool:
        return isinstance(node, LiftableDependencyTreeNode)

    def case_lifted(self, table: Table):
        return self.lifted(table)

    def lifted(self, table: Table) -> str:
        return self.word

    def nodes(self) -> List[LiftableDependencyTreeNode]:
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
        return self.parent == 0 and self.dependency == "ROOT"

    def has_parent(self) -> bool:
        return isinstance(self.parent, LiftableDependencyTreeNode)

    def has_ancestor_with_property(self, property_function: Callable[[LiftableDependencyTreeNode], bool]) -> bool:
        return any([property_function(ancestor) for ancestor in self.get_ancestors()])

    def get_ancestors(self) -> List[LiftableDependencyTreeNode]:
        res = []
        if self.has_parent():
            res.append(self.parent)
            res.extend(self.parent.get_ancestors())
        return res

    def get_position(self) -> int:
        return self.node_id

    def get_word(self) -> str:
        return self.word

    def __eq__(self, other) -> bool:
        res = False
        if LiftableDependencyTreeNode.isinstance(other):
            node_id_equals = self.node_id == other.node_id
            dependency_equals = self.dependency == other.dependency
            word_equals = self.get_word() == other.get_word()
            lifted_equals = self.lifted(None) == other.lifted(None)
            children_equals = self.children == other.children
            depth_equals = self.depth == other.depth
            parent_equals = self.parent == other.parent
            res = node_id_equals and dependency_equals and word_equals and children_equals and depth_equals and \
                  parent_equals and lifted_equals
        return res

    def __gt__(self, other) -> bool:
        return self.node_id > other.node_id

    def __repr__(self) -> str:
        return f"{self.node_id}, {self.word_type}, {self.dependency}, {self.lifted(None)}, {self.get_word()}"
