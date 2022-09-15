import copy
from copy import deepcopy
from typing import List, Tuple

from src.entity_abstractor.dependencytree.LiftableDependencyTree import LiftableDependencyTree
from src.entity_abstractor.dependencytree.creation.LiftableDependencyTreeNodeFactory \
    import LiftableDependencyTreeNodeFactory
from src.entity_abstractor.dependencytree.nodes.LiftableDependencyTreeNode import LiftableDependencyTreeNode
from src.entity_abstractor.dependencytree.nodes.LiftableDependencyTreeRootNode import LiftableDependencyTreeRootNode


class LiftableDependencyTreeCreator:
    """
    This class creates dependency trees from node data tuples.
    """
    def __init__(self, raw_nodes: List[Tuple[str, str, int, str]],
                 node_factory: LiftableDependencyTreeNodeFactory):
        self.raw_nodes = raw_nodes
        self.node_factory = node_factory

    def create_subsentence_trees_of(self, sentence_tree: LiftableDependencyTree) -> List[LiftableDependencyTree]:
        """
        If there are subsentences inside the input utterance, create subsentence dependency trees.
        :param sentence_tree: The sentence tree of the original utterance.
        :return subsentence_trees: A list of subsentence dependency trees.
        """
        subsentence_trees = [copy.deepcopy(sentence_tree)]
        for child in sentence_tree.root.children:
            if LiftableDependencyTreeRootNode.isinstance(child):
                subsentence_trees[0], subsentence_tree = self.create_subsentence_tree_from(subsentence_trees[0], child)
                subsentence_trees.append(subsentence_tree)
        return subsentence_trees

    def create_subsentence_tree_from(self,
                                     parent_sentence_tree: LiftableDependencyTree,
                                     root_node: LiftableDependencyTreeRootNode) \
            -> Tuple[LiftableDependencyTree, LiftableDependencyTree]:
        """
        Create a subsentence tree from the parent sentence and the subsentence root node.
        :param parent_sentence_tree:
        :param root_node:
        :return:
        """
        node_copy = deepcopy(root_node)
        parent_sentence_subtree = parent_sentence_tree - LiftableDependencyTree(root_node)
        node_copy = self.transform_to_sentence_root(node_copy)
        return parent_sentence_subtree, LiftableDependencyTree(node_copy)

    def transform_to_sentence_root(self, node: LiftableDependencyTreeRootNode) -> LiftableDependencyTreeRootNode:
        """
        Transform a node inside a sentence tree to a root node.
        :param node:
        :return:
        """
        node.parent = 0
        node.dependency = self.node_factory.root_token
        nodes = node.nodes()
        for new_node_id, sub_node in enumerate(nodes):
            sub_node.node_id = new_node_id
            sub_node.depth = sub_node.depth - node.depth
        node.depth = 0
        return node

    def create_tree(self) -> LiftableDependencyTree:
        root_node = self.create_root_node()
        return self.build_liftable_dependency_tree_from(root_node)

    def create_root_node(self) -> LiftableDependencyTreeNode:
        for node_id, (word, word_type, parent_id, dependency) in enumerate(self.raw_nodes):
            if parent_id == 0 and self.node_factory.is_root_dependency(dependency):
                return self.node_factory.create_initial_root_node((node_id + 1, word, word_type, parent_id, dependency))

    def build_liftable_dependency_tree_from(self, root_node: LiftableDependencyTreeNode) -> LiftableDependencyTree:
        root_node = self.create_subtree(root_node)
        return LiftableDependencyTree(root_node)

    def create_subtree(self, root_node: LiftableDependencyTreeNode) -> LiftableDependencyTreeNode:
        root_node = self.create_children(root_node)
        self.create_child_subtrees(root_node)
        return root_node

    def create_children(self, parent_node: LiftableDependencyTreeNode) -> LiftableDependencyTreeNode:
        """
        Create the child nodes for a given parent node from the node data tuples.
        :param parent_node:
        :return:
        """
        child_depth = parent_node.depth + 1
        for conll_line_index in range(len(self.raw_nodes)):
            parent_node = self.if_possible_create_child(parent_node, conll_line_index, child_depth)
        return parent_node

    def create_child_subtrees(self, parent_node: LiftableDependencyTreeNode) -> None:
        for i, child in enumerate(parent_node.children):
            parent_node.children[i] = self.create_subtree(child)

    def if_possible_create_child(self, parent_node: LiftableDependencyTreeNode,
                                 conll_line_index: int,
                                 child_depth: int) -> LiftableDependencyTreeNode:
        """
        Create child if parent node has one.
        :param parent_node:
        :param conll_line_index:
        :param child_depth: The depth of the child node inside the dependency tree.
        :return:
        """
        node_id = conll_line_index + 1
        word, word_type, parent_id, dependency = self.raw_nodes[conll_line_index]
        if parent_id == parent_node.node_id:
            parent_node = self.create_child(parent_node, (node_id, word, word_type, parent_id, dependency, child_depth))
        return parent_node

    def create_child(self, parent_node: LiftableDependencyTreeNode,
                     node_data: Tuple[int, str, str, int, str, int]) -> LiftableDependencyTreeNode:
        child, parent_node = self.node_factory.create_node(parent_node, node_data)
        parent_node.add_child(child)
        return parent_node
