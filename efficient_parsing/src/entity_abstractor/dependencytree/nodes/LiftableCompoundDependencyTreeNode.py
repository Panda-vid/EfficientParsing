from src.entity_abstractor.dependencytree.nodes.LiftableDependencyTreeNode import LiftableDependencyTreeNode


class LiftableCompoundDependencyTreeNode(LiftableDependencyTreeNode):
    @classmethod
    def isinstance(cls, node: LiftableDependencyTreeNode) -> bool:
        return isinstance(node, LiftableCompoundDependencyTreeNode)

    def lifted(self, table=None) -> str:
        return ""
