from typing import List

from smart_ccg.sem_parser.entity_abstractor.liftable_dependendency_tree.LiftableDependencyTree import \
    LiftableDependencyTree
from smart_ccg.sem_parser.entity_abstractor.liftable_dependendency_tree.nodes.LiftableCaseDependencyTreeNode import \
    LiftableCaseDependencyTreeNode
from smart_ccg.sem_parser.entity_abstractor.liftable_dependendency_tree.nodes.LiftableObjectDependencyTreeNode import \
    LiftableObjectDependencyTreeNode


class Sentence:
    def __init__(self, dependency_tree: LiftableDependencyTree,
                 objects: List[LiftableObjectDependencyTreeNode],
                 cases: List[LiftableCaseDependencyTreeNode]):
        self.dependency_tree = dependency_tree
        self.objects = objects
        self.cases = cases

    @classmethod
    def create_sentence(cls, dependency_tree: LiftableDependencyTree):
        objects, cases = dependency_tree.find_all_objects_and_cases()
        return cls(dependency_tree, objects, cases)
