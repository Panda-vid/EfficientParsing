from typing import List, Tuple

from smart_ccg.sem_parser.entity_abstractor.liftable_dependendency_tree.LiftableDependencyTree import LiftableDependencyTree
from smart_ccg.sem_parser.entity_abstractor.liftable_dependendency_tree.nodes.LiftableDependencyTreeNode import LiftableDependencyTreeNode
from smart_ccg.sem_parser.entity_abstractor.liftable_dependendency_tree.creation import LiftableDependencyTreeNodeFactory


class LiftableDependencyTreeCreator:
    def __init__(self, raw_nodes: List[Tuple[str, str, int, str]],
                 node_factory: LiftableDependencyTreeNodeFactory):
        self.raw_nodes = raw_nodes
        self.node_factory = node_factory

    @classmethod
    def create_liftable_dependency_tree(cls, raw_nodes: List[Tuple[str, str, int, str]]):
        creator = cls(raw_nodes, LiftableDependencyTreeNodeFactory.get_default_instance())
        return creator.create_tree()

    def create_tree(self):
        root_node = self.create_root_node()
        self.build_liftable_dependency_tree_from(root_node)

    def create_root_node(self):
        for word, word_type, parent_id, dependency in self.raw_nodes:
            if parent_id == 0 and self.node_factory.is_root_dependency(dependency):
                return self.node_factory.create_initial_root_node((word, word_type, parent_id, dependency))

    def build_liftable_dependency_tree_from(self, root_node: LiftableDependencyTreeNode):
        self.create_subtree(root_node)
        return LiftableDependencyTree(root_node)

    def create_subtree(self, root_node: LiftableDependencyTreeNode) -> LiftableDependencyTreeNode:
        self.if_possible_create_children(root_node)
        self.create_child_subtrees(root_node)
        return root_node

    def if_possible_create_children(self, parent_node: LiftableDependencyTreeNode) -> None:
        child_depth = parent_node.depth + 1
        for conll_line_index in range(len(self.raw_nodes)):
            self.if_possible_create_child(parent_node, conll_line_index, child_depth)

    def create_child_subtrees(self, parent_node: LiftableDependencyTreeNode) -> None:
        for child in parent_node.children:
            self.create_subtree(child)

    def if_possible_create_child(self, parent_node: LiftableDependencyTreeNode,
                                 conll_line_index: int,
                                 child_depth: int):
        node_id = conll_line_index + 1
        word, word_type, parent_id, dependency = self.raw_nodes[conll_line_index]
        if parent_id == parent_node.node_id:
            self.create_child(parent_node, (node_id, word, word_type, parent_id, dependency, child_depth))

    def create_child(self, parent_node: LiftableDependencyTreeNode, node_data: Tuple[int, str, str, int, str, int]):
        child = self.node_factory.create_node(parent_node, node_data)
        parent_node.add_child(child)
