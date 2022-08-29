from __future__ import annotations

from typing import List, Tuple

from smart_ccg.sem_parser.abstractor.dependencytree.nodes.LiftableCaseDependencyTreeNode import \
    LiftableCaseDependencyTreeNode
from smart_ccg.sem_parser.abstractor.dependencytree.nodes.LiftableDependencyTreeNode import LiftableDependencyTreeNode
from smart_ccg.sem_parser.abstractor.dependencytree.nodes.LiftableObjectDependencyTreeNode import \
    LiftableObjectDependencyTreeNode
from smart_ccg.sem_parser.abstractor.dependencytree.nodes.LiftableValueDependencyTreeNode import \
    LiftableValueDependencyTreeNode


class LiftableDependencyTree:
    def __init__(self, root: LiftableDependencyTreeNode):
        self.root = root

    def find_all_important_nodes(self) -> Tuple[List[LiftableObjectDependencyTreeNode],
                                                List[LiftableCaseDependencyTreeNode],
                                                List[LiftableValueDependencyTreeNode]]:
        return self.find_all_objects(), self.find_all_cases(), self.find_all_values()

    def find_all_objects(self) -> List[LiftableObjectDependencyTreeNode]:
        return [node for node in self.nodes() if isinstance(node, LiftableObjectDependencyTreeNode)]

    def find_all_cases(self) -> List[LiftableCaseDependencyTreeNode]:
        return [node for node in self.nodes() if isinstance(node, LiftableCaseDependencyTreeNode)]

    def find_all_values(self) -> List[LiftableValueDependencyTreeNode]:
        return [node for node in self.nodes() if isinstance(node, LiftableValueDependencyTreeNode)]

    def as_conll(self) -> str:
        return "\n".join(["{0} {1} {2} {3} {4} {5}"
                         .format(node.node_id, node.word, node.lifted, node.word_type, node.parent, node.dependency)
                          for node in self.nodes()])

    def nodes(self) -> List[LiftableDependencyTreeNode]:
        return self.root.nodes()

    def __eq__(self, other) -> bool:
        res = False
        if isinstance(other, LiftableDependencyTree):
            res = self.root == other.root
        return res
