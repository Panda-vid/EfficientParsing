from __future__ import annotations

from typing import List


class LiftableDependencyTreeNode:
    OBJECT_DEPENDENCY = "object"
    CONJUNCTION_DEPENDENCY = "conjunction"
    OBLIQUE_DEPENDENCY = "oblique"

    def __init__(self, node_id: int,
                 word: str,
                 word_type: str,
                 parent_id: int,
                 dependency: str,
                 depth=0,
                 lifted="",
                 children=None):
        if lifted == "":
            lifted = word
        if children is None:
            children = []

        self.node_id = node_id
        self.dependency = dependency
        self.word = word
        self.lifted = lifted
        self.word_type = word_type
        self.children = children
        self.depth = depth
        self.parent = parent_id

    def lift_subtree(self, object_dependency_tokens: List[str],
                     oblique_dependency_tokens: List[str],
                     conjunction_tokens: List[str]) -> None:
        for child in self.children:
            self.decide_child_dependency_and_lift_node(child, object_dependency_tokens, oblique_dependency_tokens,
                                                  conjunction_tokens)
        self.resolve_neighboring_object_nodes()

    def decide_child_dependency_and_lift_node(self, child: LiftableDependencyTreeNode,
                                              object_dependency_tokens: List[str],
                                              oblique_dependency_tokens: List[str],
                                              conjunction_tokens: List[str]):
        if child.dependency in object_dependency_tokens:
            child.lift_object_node(object_dependency_tokens, conjunction_tokens)
        if child.dependency in conjunction_tokens:
            child.lift_subtree(object_dependency_tokens, oblique_dependency_tokens, conjunction_tokens)
        if child.dependency in oblique_dependency_tokens:
            child.lift_oblique_node()

    def lift_object_node(self, object_dependency_tokens: List[str], conjunction_tokens: List[str]) -> None:
        self.lifted = "<obj>"
        for child in self.children:
            if child.dependency in object_dependency_tokens or child.dependency in conjunction_tokens:
                child.lift_object_node(object_dependency_tokens, conjunction_tokens)
            elif child.dependency == "<case>":
                child.lift_case()
            else:
                child.update_subtree_to_lifted()

    def lift_oblique_node(self) -> None:
        for child in self.children:
            if child.dependency == "case":
                self.lift_case()
            if child.dependency == "nummod":
                self.lift_value()

    def lift_value(self) -> None:
        self.lifted = ""
        for child in self.children:
            if child.dependency == "nummod":
                child.lifted = "<val>"

    def lift_case(self) -> None:
        nodes = self.nodes()
        nodes[0].lifted = "[" + nodes[0].lifted
        nodes[-1].lifted += "]"
        if self.word_type == "NN":
            self.lifted = "<obj>"
        self.update_subtree_to_lifted()
        for child in self.children:
            if child.dependency == "obl":
                child.lift_oblique_node()
            if child.dependency == "nmod":
                child.lift_value()
            if child.dependency == "amod":
                child.lift_comparison_rhs()

    def lift_comparison_rhs(self) -> None:
        for child in self.children:
            if child.dependency == "obl":
                child.lift_oblique_node()
            if child.dependency == "nmod":
                child.lift_value()

    def update_subtree_to_lifted(self, removal_dependencies=None) -> None:
        if removal_dependencies is None:
            removal_dependencies = ["det", "cc", "punct"]
        if self.dependency in removal_dependencies:
            self.lifted = ""
        for child in self.children:
            child.update_subtree_to_lifted(removal_dependencies)

    def resolve_neighboring_object_nodes(self) -> None:
        pack = []
        for node in self.lifted_nodes():
            if node.lifted == "<obj>":
                pack.append(node)
            else:
                if len(pack) > 1:
                    pack[0].lifted = "<objs>[{0}]".format(", ".join([str(pack_node.node_id) for pack_node in pack]))
                    for pack_node in pack[1:]:
                        pack_node.lifted = ""
                pack = []

    def transform_to_sentence_root(self) -> None:
        self.parent = 0
        self.dependency = "ROOT"
        nodes = self.nodes()
        for new_node_id, node in enumerate(nodes):
            node.node_id = new_node_id
        self.update_children_parent_ids()

    def update_children_parent_ids(self) -> None:
        for child in self.children:
            child.parent = self.node_id
            if not child.is_leaf():
                child.update_children_parent_ids()

    def lifted_nodes(self) -> List[LiftableDependencyTreeNode]:
        return [node for node in self.nodes() if node.lifted != ""]

    def nodes(self) -> List[LiftableDependencyTreeNode]:
        res = [self]
        for child in self.children:
            if child.is_leaf():
                res.append(child)
            else:
                res += child.nodes()
        res.sort()
        return res

    def get_parent_dependency(self):
        parent_dependency = None
        if self.has_parent():
            parent_dependency = self.parent.dependency
        return parent_dependency

    def get_child_dependencies(self):
        return [child.dependency for child in self.children]

    def add_child(self, child: LiftableDependencyTreeNode) -> None:
        self.children.append(child)
        child.set_parent(self)

    def set_parent(self, parent: LiftableDependencyTreeNode) -> None:
        self.parent = parent

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def is_sentence_root(self) -> bool:
        return self.parent == 0 and self.dependency == "ROOT"

    def is_object_node(self) -> bool:
        return self.dependency == LiftableDependencyTreeNode.OBJECT_DEPENDENCY

    def is_case_node(self) -> bool:
        return self.dependency == LiftableDependencyTreeNode.OBLIQUE_DEPENDENCY \
               and any([child.dependency == "case" for child in self.children])

    def is_conjunction_node(self) -> bool:
        return self.dependency == LiftableDependencyTreeNode.CONJUNCTION_DEPENDENCY

    def has_parent(self):
        return isinstance(self.parent, LiftableDependencyTreeNode)

    def __eq__(self, other) -> bool:
        res = False
        if isinstance(other, LiftableDependencyTreeNode):
            node_id_equals = self.node_id == other.node_id
            dependency_equals = self.dependency == other.dependency
            word_equals = self.word == other.word
            lifted_equals = self.lifted == other.lifted
            children_equals = self.children == other.children
            depth_equals = self.depth == other.depth
            parent_equals = self.parent == other.parent
            res = node_id_equals and dependency_equals and word_equals and children_equals and depth_equals and \
                  parent_equals and lifted_equals
        return res

    def __gt__(self, other) -> bool:
        return self.node_id > other.node_id
