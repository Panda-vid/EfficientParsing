from __future__ import annotations
from typing import List, Tuple

from smart_ccg.sem_parser.entity_abstractor.liftable_dependendency_tree.LiftableDependencyTree import LiftableDependencyTree, \
    LiftableDependencyTreeNode
from smart_ccg.sem_parser.entity_abstractor.liftable_dependendency_tree.LiftableDependencyTreeCreator import LiftableDependencyTreeCreator


class Sentence:
    def __init__(self, dependency_tree: LiftableDependencyTree):
        self.dependency_tree = dependency_tree

    @classmethod
    def create_sentnece_instance_from(cls, raw_nodes: List[Tuple[str, str, int, str]]) -> Sentence:
        sentence_tree = LiftableDependencyTreeCreator.create_liftable_dependency_tree(raw_nodes)
        return Sentence(sentence_tree)

    def lifted(self) -> Tuple[List[LiftableDependencyTreeNode], str]:
        return self.get_lifted_nodes(), self.get_lifted_string()

    def get_lifted_nodes(self) -> List[LiftableDependencyTreeNode]:
        return [node for node in self.dependency_tree.nodes() if node.lifted != ""]

    def get_lifted_string(self) -> str:
        lifted_nodes = self.get_lifted_nodes()
        return " ".join(node.lifted for node in lifted_nodes[:-1]) + lifted_nodes[-1].lifted

    def original(self) -> str:
        sentence_nodes = self.dependency_tree.nodes()
        return " ".join([node.word for node in sentence_nodes[:-1]]) + sentence_nodes[-1].word

    def as_conll(self) -> str:
        return self.dependency_tree.as_conll()
