from src.entity_abstractor.dependencytree.nodes.LiftableDependencyTreeNode import LiftableDependencyTreeNode


class LiftableDependencyTreeRootNode(LiftableDependencyTreeNode):
    def __init__(self, node_id: int, word: str, word_type: str, parent_id: int, dependency: str):
        super().__init__(node_id, word, word_type, parent_id, dependency)

    @classmethod
    def isinstance(cls, node: LiftableDependencyTreeNode) -> bool:
        return isinstance(node, LiftableDependencyTreeRootNode)

    def update_children_parent_ids(self):
        for child in self.children:
            child.parent = self.node_id
            if not child.is_leaf():
                child.update_children_parent_ids()
