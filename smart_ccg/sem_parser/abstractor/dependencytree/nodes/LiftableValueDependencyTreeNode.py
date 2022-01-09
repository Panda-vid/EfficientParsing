from __future__ import annotations

from smart_ccg.sem_parser.entity_abstractor.liftable_dependendency_tree.nodes.LiftableDependencyTreeNode import \
    LiftableDependencyTreeNode


class LiftableValueDependencyTreeNode(LiftableDependencyTreeNode):
    def __init__(self, node_id: int, word: str, word_type: str, parent_id: int, dependency: str, depth: int):
        super().__init__(node_id, word, word_type, parent_id, dependency, depth)

    def get_position(self) -> int:
        return self.node_id

    def get_word(self) -> str:
        return self.word