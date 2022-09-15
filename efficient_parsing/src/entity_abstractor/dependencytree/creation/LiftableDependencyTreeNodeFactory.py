from typing import Tuple, List

from src.entity_abstractor.dependencytree.nodes.LiftableCaseDependencyTreeNode import LiftableCaseDependencyTreeNode
from src.entity_abstractor.dependencytree.nodes.LiftableCompoundDependencyTreeNode import \
    LiftableCompoundDependencyTreeNode
from src.entity_abstractor.dependencytree.nodes.LiftableDependencyTreeNode import LiftableDependencyTreeNode
from src.entity_abstractor.dependencytree.nodes.LiftableDependencyTreeRootNode import LiftableDependencyTreeRootNode
from src.entity_abstractor.dependencytree.nodes.LiftableObjectDependencyTreeNode import LiftableObjectDependencyTreeNode
from src.entity_abstractor.dependencytree.nodes.LiftableStopwordDependencyTreeNode import \
    LiftableStopwordDependencyTreeNode
from src.entity_abstractor.dependencytree.nodes.LiftableValueDependencyTreeNode import LiftableValueDependencyTreeNode


class LiftableDependencyTreeNodeFactory:
    """
    This class decides which dependency node type gets assigned to the words in a given sentence.
    """
    def __init__(self, object_dependency_tokens: List[str],
                 value_dependency_tokens: List[str],
                 stopword_dependency_tokens: List[str],
                 oblique_dependency_token: str,
                 compound_dependency_token: str,
                 conjunction_dependency_token: str,
                 case_dependency_token: str,
                 auxilliary_dependency_token: str,
                 root_token: str):
        """

        :param object_dependency_tokens: List of universal dependencies corresponding to object nodes.
        :param value_dependency_tokens: List of universal dependencies corresponding to value nodes.
        :param stopword_dependency_tokens: List of universal dependencies corresponding to stopword nodes.
        :param oblique_dependency_token: List of universal dependencies corresponding to oblique dependencies.
        :param compound_dependency_token: List of universal dependencies corresponding to compound nodes.
        :param conjunction_dependency_token: List of universal dependencies corresponding to conjunctions.
        :param case_dependency_token: List of universal dependencies corresponding to case nodes.
        :param auxilliary_dependency_token: List of universal dependencies corresponding to auxilliary dependencies.
        :param root_token:
        """
        self.object_dependency_tokens = object_dependency_tokens
        self.value_dependency_tokens = value_dependency_tokens
        self.stopword_dependency_tokens = stopword_dependency_tokens
        self.oblique_dependency_token = oblique_dependency_token
        self.compound_dependency_token = compound_dependency_token
        self.conjunction_dependency_token = conjunction_dependency_token
        self.case_dependency_token = case_dependency_token
        self.auxilliary_dependency_token = auxilliary_dependency_token
        self.root_token = root_token

    @classmethod
    def get_default_instance(cls):
        """
        Create a Factory using the default lists of dependencies.
        :return:
        """
        object_dependency_tokens = ["obj", "appos", "nmod"]
        value_dependency_tokens = ["nummod", "amod", "acl:relcl"]
        stopword_dependency_tokens = ["det", "nsubj", "punct", "cc", "acl"]
        oblique_dependency_token = "obl"
        compound_dependency_token = "compound"
        conjunction_dependency_token = "conj"
        case_dependency_token = "case"
        auxilliary_dependency_token = "aux"
        root_token = "ROOT"
        return cls(object_dependency_tokens,
                   value_dependency_tokens,
                   stopword_dependency_tokens,
                   oblique_dependency_token,
                   compound_dependency_token,
                   conjunction_dependency_token,
                   case_dependency_token,
                   auxilliary_dependency_token,
                   root_token)

    @classmethod
    def create_initial_root_node(cls, root_node_data: Tuple[int, str, str, int, str]) \
            -> LiftableDependencyTreeRootNode:
        node_id, word, word_type, parent_id, dependency = root_node_data
        dependency = "ROOT"
        return LiftableDependencyTreeRootNode(node_id, word, word_type, parent_id, dependency)

    def create_node(self, parent_node: LiftableDependencyTreeNode, node_data: Tuple[int, str, str, int, str, int])\
            -> Tuple[LiftableDependencyTreeNode, LiftableDependencyTreeNode]:
        """
        Create a node and give the correct type according to the dependency parse structure.
        :param parent_node:
        :param node_data:
        :return:
        """
        node_id, word, word_type, parent_id, dependency, depth = node_data
        if self.is_object_dependency(dependency):
            node = LiftableObjectDependencyTreeNode(node_id, word, word_type, parent_id, dependency, depth)
        elif self.is_oblique_case_dependency(parent_node, dependency):
            parent_node = LiftableObjectDependencyTreeNode(parent_node.node_id, parent_node.word, parent_node.word_type,
                                                           parent_node.parent, parent_node.dependency,
                                                           parent_node.depth)
            node = LiftableCaseDependencyTreeNode(node_id, word, word_type, parent_id, dependency, depth)
        elif self.is_object_case_dependency(parent_node, dependency):
            node = LiftableCaseDependencyTreeNode(node_id, word, word_type, parent_id, dependency, depth)
        elif self.is_value_dependency(parent_node.dependency, dependency):
            node = LiftableValueDependencyTreeNode(node_id, word, word_type, parent_id, dependency, depth)
        elif self.is_stopword_dependency(dependency):
            node = LiftableStopwordDependencyTreeNode(node_id, word, word_type, parent_id, dependency, depth)
        elif self.is_compound_dependency(dependency):
            node = LiftableCompoundDependencyTreeNode(node_id, word, word_type, parent_id, dependency, depth)
        elif self.is_auxilliary_dependency(dependency):
            node = self.resolve_auxilliary_dependency(parent_node, node_id, word, word_type, parent_id, dependency,
                                                      depth)
        elif self.is_conjunction_dependency(dependency):
            node = self.resolve_conjunction_dependency(parent_node, node_id, word, word_type, parent_id, dependency,
                                                       depth)
        else:
            node = LiftableDependencyTreeNode(node_id, word, word_type, parent_id, dependency, depth)
        return node, parent_node

    def resolve_auxilliary_dependency(self, parent_node: LiftableDependencyTreeNode,
                                      node_id: int, word: str, word_type: str, parent_id: int, dependency: str,
                                      depth: int) -> LiftableDependencyTreeNode:
        """
        Find correct node type for a word with auxilliary dependency.
        This can either be a case node if the parent node is a value or a stopword if the parent is not a value node.
        :param parent_node:
        :param node_id:
        :param word:
        :param word_type:
        :param parent_id:
        :param dependency:
        :param depth:
        :return:
        """
        if LiftableValueDependencyTreeNode.isinstance(parent_node):
            node = LiftableCaseDependencyTreeNode(node_id, word, word_type, parent_id, dependency, depth)
        else:
            node = LiftableStopwordDependencyTreeNode(node_id, word, word_type, parent_id, dependency, depth)
        return node

    def resolve_conjunction_dependency(self, parent_node: LiftableDependencyTreeNode,
                                       node_id: int, word: str, word_type: str, parent_id: int, dependency: str,
                                       depth: int) -> LiftableDependencyTreeNode:
        """
        Inherit the node type of the conjunction head in the dependency parse.
        :param parent_node:
        :param node_id:
        :param word:
        :param word_type:
        :param parent_id:
        :param dependency:
        :param depth:
        :return:
        """
        if LiftableObjectDependencyTreeNode.isinstance(parent_node):
            node = LiftableObjectDependencyTreeNode(node_id, word, word_type, parent_id, dependency, depth)
        elif LiftableCaseDependencyTreeNode.isinstance(parent_node):
            node = LiftableCaseDependencyTreeNode(node_id, word, word_type, parent_id, dependency, depth)
        elif LiftableDependencyTreeRootNode.isinstance(parent_node):
            node = LiftableDependencyTreeRootNode(node_id, word, word_type, parent_id, dependency)
        else:
            node = LiftableDependencyTreeNode(node_id, word, word_type, parent_id, dependency, depth)
        return node

    def is_object_dependency(self, dependency: str) -> bool:
        return dependency in self.object_dependency_tokens

    def is_oblique_case_dependency(self, parent: LiftableDependencyTreeNode, dependency: str) -> bool:
        return self.is_oblique_dependency(parent.dependency) and self.is_case_dependency(dependency)

    def is_object_case_dependency(self, parent: LiftableDependencyTreeNode, dependency: str) -> bool:
        return self.is_object_dependency(parent.dependency) and self.is_case_dependency(dependency)

    def is_stopword_dependency(self, dependency: str) -> bool:
        return dependency in self.stopword_dependency_tokens

    def is_conjunction_dependency(self, dependency: str) -> bool:
        return dependency == self.conjunction_dependency_token

    def is_value_dependency(self, parent_dependency, dependency: str) -> bool:
        return dependency in self.value_dependency_tokens or \
               (self.is_oblique_dependency(dependency) and self.is_case_dependency(parent_dependency))

    def is_root_dependency(self, dependency: str) -> bool:
        return dependency == self.root_token

    def is_auxilliary_dependency(self, dependency: str) -> bool:
        return dependency == self.auxilliary_dependency_token

    def is_oblique_dependency(self, dependency: str) -> bool:
        return dependency == self.oblique_dependency_token

    def is_compound_dependency(self, dependency: str) -> bool:
        return dependency == self.compound_dependency_token

    def is_case_dependency(self, dependency: str) -> bool:
        return dependency == self.case_dependency_token
