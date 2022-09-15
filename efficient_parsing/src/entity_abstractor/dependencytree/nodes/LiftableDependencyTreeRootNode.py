from src.entity_abstractor.dependencytree.nodes.LiftableDependencyTreeNode import LiftableDependencyTreeNode


class LiftableDependencyTreeRootNode(LiftableDependencyTreeNode):
    """
    This class represents root nodes of dependency trees.
    """
    def __init__(self, node_id: int, word: str, word_type: str, parent_id: int, dependency: str):
        super().__init__(node_id, word, word_type, parent_id, dependency)

    @classmethod
    def isinstance(cls, node: LiftableDependencyTreeNode) -> bool:
        """
        Check whether input node is a root node.
        :param node:
        :return:
        """
        return isinstance(node, LiftableDependencyTreeRootNode)

    def update_children_parent_ids(self) -> None:
        """
        Update all child ids if this root is a subsentence root such that the node ids start from 0 again.
        :return:
        """
        for child in self.children:
            child.parent = self.node_id
            if not child.is_leaf():
                child.update_children_parent_ids()
