from __future__ import annotations

import copy
from typing import List, Tuple

from smart_ccg.sem_parser.entity_abstractor.liftable_dependendency_tree.nodes.LiftableDependencyTreeNode import LiftableDependencyTreeNode
from smart_ccg.sem_parser.entity_abstractor.liftable_dependendency_tree.nodes.LiftableCaseDependencyTreeNode import LiftableCaseDependencyTreeNode
from smart_ccg.sem_parser.entity_abstractor.liftable_dependendency_tree.nodes.LiftableObjectDependencyTreeNode import LiftableObjectDependencyTreeNode


class LiftableDependencyTree:
    def __init__(self, root: LiftableDependencyTreeNode):
        self.root = root

    def get_subsentence_dependency_trees(self) -> List[LiftableDependencyTree]:
        return [LiftableDependencyTree(child.transform_to_sentence_root())
                for child in copy.deepcopy(self.root.children)
                if child.dependency == LiftableDependencyTreeNode.CONJUNCTION_DEPENDENCY]

    def find_all_objects_and_cases(self) -> Tuple[List[LiftableObjectDependencyTreeNode],
                                                  List[LiftableCaseDependencyTreeNode]]:
        return self.find_all_objects(), self.find_all_cases()

    def find_all_objects(self) -> List[LiftableObjectDependencyTreeNode]:
        return [node for node in self.nodes() if isinstance(node, LiftableObjectDependencyTreeNode)]

    def find_all_cases(self) -> List[LiftableCaseDependencyTreeNode]:
        return [node for node in self.nodes() if isinstance(node, LiftableCaseDependencyTreeNode)]

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
