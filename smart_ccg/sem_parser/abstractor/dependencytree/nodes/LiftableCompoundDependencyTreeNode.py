from smart_ccg.sem_parser.abstractor.Table import Table
from smart_ccg.sem_parser.abstractor.dependencytree.nodes.LiftableDependencyTreeNode import LiftableDependencyTreeNode


class LiftableCompoundDependencyTreeNode(LiftableDependencyTreeNode):
    @classmethod
    def isinstance(cls, node: LiftableDependencyTreeNode) -> bool:
        return isinstance(node, LiftableCompoundDependencyTreeNode)

    def lifted(self, table: Table = None) -> str:
        return ""
