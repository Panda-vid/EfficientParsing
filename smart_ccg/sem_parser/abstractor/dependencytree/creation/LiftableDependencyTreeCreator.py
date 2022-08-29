from copy import deepcopy
from typing import List, Tuple

from smart_ccg.sem_parser.abstractor.dependencytree.LiftableDependencyTree import LiftableDependencyTree
from smart_ccg.sem_parser.abstractor.dependencytree.creation.LiftableDependencyTreeNodeFactory import \
    LiftableDependencyTreeNodeFactory
from smart_ccg.sem_parser.abstractor.dependencytree.nodes.LiftableDependencyTreeNode import LiftableDependencyTreeNode
from smart_ccg.sem_parser.abstractor.dependencytree.nodes.LiftableDependencyTreeRootNode import \
    LiftableDependencyTreeRootNode


class LiftableDependencyTreeCreator:
    def __init__(self, raw_nodes: List[Tuple[str, str, int, str]],
                 node_factory: LiftableDependencyTreeNodeFactory):
        self.raw_nodes = raw_nodes
        self.node_factory = node_factory

    def create_subsentence_trees_of(self, sentence_tree: LiftableDependencyTree) -> List[LiftableDependencyTree]:
        return [self.create_subsentence_tree_from(child) for child in sentence_tree.root.children
                if isinstance(child, LiftableDependencyTreeRootNode)]

    def create_subsentence_tree_from(self, root_node: LiftableDependencyTreeRootNode) -> LiftableDependencyTree:
        node_copy = deepcopy(root_node)
        if not node_copy.is_sentence_root():
            node_copy = self.transform_to_sentence_root(node_copy)
        return LiftableDependencyTree(node_copy)

    def transform_to_sentence_root(self, node: LiftableDependencyTreeRootNode) -> LiftableDependencyTreeRootNode:
        node.parent = 0
        node.dependency = self.node_factory.root_token
        nodes = node.nodes()
        for new_node_id, sub_node in enumerate(nodes):
            sub_node.node_id = new_node_id
        node.update_children_parent_ids()
        return node

    def create_tree(self) -> LiftableDependencyTree:
        root_node = self.create_root_node()
        return self.build_liftable_dependency_tree_from(root_node)

    def create_root_node(self) -> LiftableDependencyTreeNode:
        for node_id, (word, word_type, parent_id, dependency) in enumerate(self.raw_nodes):
            if parent_id == 0 and self.node_factory.is_root_dependency(dependency):
                return self.node_factory.create_initial_root_node((node_id + 1, word, word_type, parent_id, dependency))

    def build_liftable_dependency_tree_from(self, root_node: LiftableDependencyTreeNode) -> LiftableDependencyTree:
        self.create_subtree(root_node)
        return LiftableDependencyTree(root_node)

    def create_subtree(self, root_node: LiftableDependencyTreeNode) -> LiftableDependencyTreeNode:
        self.create_children(root_node)
        self.create_child_subtrees(root_node)
        return root_node

    def create_children(self, parent_node: LiftableDependencyTreeNode) -> None:
        child_depth = parent_node.depth + 1
        for conll_line_index in range(len(self.raw_nodes)):
            self.if_possible_create_child(parent_node, conll_line_index, child_depth)

    def create_child_subtrees(self, parent_node: LiftableDependencyTreeNode) -> None:
        for child in parent_node.children:
            self.create_subtree(child)

    def if_possible_create_child(self, parent_node: LiftableDependencyTreeNode,
                                 conll_line_index: int,
                                 child_depth: int) -> None:
        node_id = conll_line_index + 1
        word, word_type, parent_id, dependency = self.raw_nodes[conll_line_index]
        if parent_id == parent_node.node_id:
            self.create_child(parent_node, (node_id, word, word_type, parent_id, dependency, child_depth))

    def create_child(self, parent_node: LiftableDependencyTreeNode,
                     node_data: Tuple[int, str, str, int, str, int]) -> None:
        child = self.node_factory.create_node(parent_node, node_data)
        parent_node.add_child(child)
