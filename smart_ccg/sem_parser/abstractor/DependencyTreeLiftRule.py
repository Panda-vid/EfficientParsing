from __future__ import annotations
from typing import Callable

from smart_ccg.sem_parser.entity_abstractor.liftable_dependendency_tree.nodes.LiftableDependencyTreeNode import LiftableDependencyTreeNode


class DependencyTreeLiftRule:
    def __init__(self, parent_dependency: str,
                 node_dependency: str,
                 child_dependency: str,
                 on_apply: Callable[[LiftableDependencyTreeNode], None]):
        self.parent_dependency = parent_dependency
        self.node_dependency = node_dependency
        self.child_dependency = child_dependency
        self.on_apply = on_apply

    @classmethod
    def empty_rule(cls) -> DependencyTreeLiftRule:
        return cls("", "", "", lambda node: None)

    def apply(self, node):
        self.on_apply(node)

    def is_applicable(self, node: LiftableDependencyTreeNode):
        return self.has_dependency_or_any_dependency(node) \
               and self.has_parent_dependency_or_any_parent(node) \
               and self.has_child_dependency_or_any_children(node)

    def has_parent_dependency_or_any_parent(self, node: LiftableDependencyTreeNode) -> bool:
        return self.parent_dependency == node.get_parent_dependency() or self.parent_dependency == ""

    def has_child_dependency_or_any_children(self, node: LiftableDependencyTreeNode) -> bool:
        return self.child_dependency in node.get_child_dependencies() or self.child_dependency == ""

    def has_dependency_or_any_dependency(self, node: LiftableDependencyTreeNode) -> bool:
        return self.node_dependency == node.dependency or self.node_dependency == ""
