from src.entity_abstractor.dependencytree.nodes.LiftableDependencyTreeNode import LiftableDependencyTreeNode


class LiftableCompoundDependencyTreeNode(LiftableDependencyTreeNode):
    """
    This class represents nodes in a dependency tree which are associated with word compounds.
    """
    @classmethod
    def isinstance(cls, node: LiftableDependencyTreeNode) -> bool:
        """
        Check whether input node is a compound node.
        :param node:
        :return:
        """
        return isinstance(node, LiftableCompoundDependencyTreeNode)

    def lifted(self, table=None) -> str:
        """
        Get the representation of the word corresponding to this node in the lifted string.
        :param table: The active table in the parser's context.
        :return:
        """
        return ""
