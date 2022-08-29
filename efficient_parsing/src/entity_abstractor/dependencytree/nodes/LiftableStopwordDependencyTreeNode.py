from src.datamodel.Table import Table
from src.entity_abstractor.dependencytree.nodes.LiftableDependencyTreeNode import LiftableDependencyTreeNode


class LiftableStopwordDependencyTreeNode(LiftableDependencyTreeNode):
    def __init__(self, node_id: int, word: str, word_type: str, parent_id: int, dependency: str, depth: int):
        super().__init__(node_id, word, word_type, parent_id, dependency, depth=depth)
        self.lifted_string = ""

    @classmethod
    def isinstance(cls, node: LiftableDependencyTreeNode) -> bool:
        return isinstance(node, LiftableStopwordDependencyTreeNode)

    def case_lifted(self, table=None):
        return self.lifted(table)

    def lifted(self, table=None):
        return self.lifted_string
