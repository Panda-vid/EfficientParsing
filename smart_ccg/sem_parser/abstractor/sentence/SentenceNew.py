from __future__ import annotations

from typing import List

from smart_ccg.sem_parser.abstractor.dependencytree.LiftableDependencyTree import LiftableDependencyTree
from smart_ccg.sem_parser.abstractor.dependencytree.nodes.LiftableCaseDependencyTreeNode import \
    LiftableCaseDependencyTreeNode
from smart_ccg.sem_parser.abstractor.dependencytree.nodes.LiftableObjectDependencyTreeNode import \
    LiftableObjectDependencyTreeNode
from smart_ccg.sem_parser.abstractor.dependencytree.nodes.LiftableValueDependencyTreeNode import \
    LiftableValueDependencyTreeNode


class Sentence:
    def __init__(self, dependency_tree: LiftableDependencyTree,
                 objects: List[LiftableObjectDependencyTreeNode],
                 cases: List[LiftableCaseDependencyTreeNode],
                 values: List[LiftableValueDependencyTreeNode]):
        self.dependency_tree = dependency_tree
        self.objects = objects
        self.cases = cases
        self.values = values

    @classmethod
    def create_sentence(cls, dependency_tree: LiftableDependencyTree) -> Sentence:
        objects, cases, values = dependency_tree.find_all_important_nodes()
        return cls(dependency_tree, objects, cases, values)

    def lifted(self, table: Table = None) -> str:
        nonempty_lifted_strings = [node.lifted(table) for node in self.dependency_tree.nodes()
                                   if node.lifted(table) != ""]
        return " ".join(nonempty_lifted_strings[:-1]) + nonempty_lifted_strings[-1]

    def case_lifted(self, table: Table = None) -> List[str]:
        return [" ".join([node.case_lifted(table)
                          for node in case.nodes() if node.case_lifted(table) != ""])
                for case in self.cases]
