from __future__ import annotations
from typing import List

from smart_ccg.sem_parser.entity_abstractor.liftable_dependendency_tree.nodes.LiftableDependencyTreeNode import \
    LiftableDependencyTreeNode
from smart_ccg.sem_parser.entity_abstractor.liftable_dependendency_tree.nodes.LiftableValueDependencyTreeNode import \
    LiftableValueDependencyTreeNode


class LiftableCaseDependencyTreeNode(LiftableDependencyTreeNode):
    def __init__(self, node_id: int, word: str, word_type: str, parent_id: int, dependency: str, depth: int):
        super().__init__(node_id, word, word_type, parent_id, dependency, depth)

    def get_case_list(self) -> List[LiftableCaseDependencyTreeNode]:
        res = [self]
        res.extend(self.get_subcases())
        return res

    def get_subcases(self) -> List[LiftableCaseDependencyTreeNode]:
        return [child for child in self.children if isinstance(child, LiftableCaseDependencyTreeNode)]

    def get_values(self) -> List[LiftableValueDependencyTreeNode]:
        return [child for child in self.children if isinstance(child, LiftableValueDependencyTreeNode)]

    def is_case_enumeration(self) -> bool:
        return any([isinstance(child, LiftableCaseDependencyTreeNode) for child in self.children])

    def get_position(self) -> int:
        return self.node_id

    def get_word(self) -> str:
        return self.word
