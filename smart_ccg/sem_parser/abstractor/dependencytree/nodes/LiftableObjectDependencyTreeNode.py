from __future__ import annotations
from typing import List

from smart_ccg.sem_parser.entity_abstractor.liftable_dependendency_tree.nodes.LiftableDependencyTreeNode import \
    LiftableDependencyTreeNode


class LiftableObjectDependencyTreeNode(LiftableDependencyTreeNode):
    def __init__(self, node_id: int, word: str, word_type: str, parent_id: int, dependency: str, depth: int):
        super().__init__(node_id, word, word_type, parent_id, dependency, depth)

    def get_object_list(self) -> List[LiftableObjectDependencyTreeNode]:
        res = [self]
        res.extend(self.get_subobjects())
        return res

    def get_subobjects(self) -> List[LiftableObjectDependencyTreeNode]:
        return [child for child in self.children if isinstance(child, LiftableObjectDependencyTreeNode)]

    def is_object_enumeration(self) -> bool:
        return any([isinstance(child, LiftableObjectDependencyTreeNode) for child in self.children])

    def get_position(self) -> int:
        return self.node_id

    def get_word(self) -> str:
        return self.word
