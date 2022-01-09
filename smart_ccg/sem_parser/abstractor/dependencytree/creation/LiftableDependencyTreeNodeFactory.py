from typing import Tuple, List

from smart_ccg.sem_parser.entity_abstractor.liftable_dependendency_tree.nodes.LiftableCaseDependencyTreeNode import \
    LiftableCaseDependencyTreeNode
from smart_ccg.sem_parser.entity_abstractor.liftable_dependendency_tree.nodes.LiftableDependencyTreeNode import LiftableDependencyTreeNode
from smart_ccg.sem_parser.entity_abstractor.liftable_dependendency_tree.nodes.LiftableObjectDependencyTreeNode import \
    LiftableObjectDependencyTreeNode
from smart_ccg.sem_parser.entity_abstractor.liftable_dependendency_tree.nodes.LiftableValueDependencyTreeNode import \
    LiftableValueDependencyTreeNode


class LiftableDependencyTreeNodeFactory:
    def __init__(self, object_dependency_tokens: List[str],
                 value_dependency_tokens: List[str],
                 oblique_dependency_token: str,
                 conjunction_dependency_token: str,
                 case_dependency_token: str,
                 root_token: str):
        self.object_dependency_tokens = object_dependency_tokens
        self.value_dependency_tokens = value_dependency_tokens
        self.oblique_dependency_token = oblique_dependency_token
        self.conjunction_dependency_token = conjunction_dependency_token
        self.case_dependency_token = case_dependency_token
        self.root_token = root_token

    @classmethod
    def get_default_instance(cls):
        object_dependency_tokens = ["obj", "appos", "nmod"]
        value_dependency_tokens = ["nummod", "amod"]
        oblique_dependency_token = "obl"
        conjunction_dependency_token = "conj"
        case_dependency_token = "case"
        root_token = "ROOT"
        return cls(object_dependency_tokens,
                   value_dependency_tokens,
                   oblique_dependency_token,
                   conjunction_dependency_token,
                   case_dependency_token,
                   root_token)

    @classmethod
    def create_initial_root_node(self, root_node_data: Tuple[str, str, int, str]):
        word, word_type, parent_id, dependency = root_node_data
        dependency = "ROOT"
        return LiftableDependencyTreeNode(1, word, word_type, parent_id, dependency)

    def create_node(self, parent_node: LiftableDependencyTreeNode, node_data: Tuple[int, str, str, int, str, int]):
        node_id, word, word_type, parent_id, dependency, depth = node_data
        if self.is_object_dependency(dependency):
            node = LiftableObjectDependencyTreeNode(node_id, word, word_type, parent_id, dependency, depth)
        elif self.is_case_dependency(parent_node.dependency, dependency):
            node = LiftableCaseDependencyTreeNode(node_id, word, word_type, parent_id, dependency, depth)
        elif self.is_value_dependency(dependency):
            node = LiftableValueDependencyTreeNode(node_id, word, word_type, parent_id, dependency, depth)
        elif self.is_conjunction_dependency(dependency):
            node = self.resolve_conjunction_node(parent_node, node_id, word, word_type, parent_id, dependency, depth)
        else:
            node = LiftableDependencyTreeNode(node_id, word, word_type, parent_id, dependency, depth)
        return node

    def resolve_conjunction_node(self, parent_node: LiftableDependencyTreeNode,
                                 node_id: int, word: str, word_type: str, parent_id: int, dependency: str, depth: int):
        if isinstance(parent_node, LiftableObjectDependencyTreeNode):
            node = LiftableObjectDependencyTreeNode(node_id, word, word_type, parent_id, dependency, depth)
        elif isinstance(parent_node, LiftableCaseDependencyTreeNode):
            node = LiftableCaseDependencyTreeNode(node_id, word, word_type, parent_id, dependency, depth)
        else:
            node = LiftableDependencyTreeNode(node_id, word, word_type, parent_id, dependency, depth)
        return node

    def is_object_dependency(self, dependency: str):
        return dependency in self.object_dependency_tokens

    def is_case_dependency(self, parent_dependency: str, dependency: str):
        return parent_dependency in self.oblique_dependency_token and dependency == self.case_dependency_token

    def is_conjunction_dependency(self, dependency: str):
        return dependency == self.conjunction_dependency_token

    def is_value_dependency(self, dependency: str):
        return dependency in self.value_dependency_tokens

    def is_root_dependency(self, dependency: str):
        return dependency == self.root_token
