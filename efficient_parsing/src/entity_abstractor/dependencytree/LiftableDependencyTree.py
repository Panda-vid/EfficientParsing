from __future__ import annotations

from typing import List, Tuple

from src.entity_abstractor.dependencytree.nodes.LiftableCaseDependencyTreeNode import LiftableCaseDependencyTreeNode
from src.entity_abstractor.dependencytree.nodes.LiftableDependencyTreeNode import LiftableDependencyTreeNode
from src.entity_abstractor.dependencytree.nodes.LiftableObjectDependencyTreeNode import LiftableObjectDependencyTreeNode
from src.entity_abstractor.dependencytree.nodes.LiftableValueDependencyTreeNode import LiftableValueDependencyTreeNode


class LiftableDependencyTree:
    """
    This class implements a dependency tree which contains nodes that can be used for entity abstraction.
    """
    def __init__(self, root: LiftableDependencyTreeNode):
        """
        :param root: The root of the dependency tree
        """
        self.root = root

    def find_all_important_nodes(self) -> Tuple[List[List[LiftableObjectDependencyTreeNode]],
                                                List[LiftableCaseDependencyTreeNode],
                                                List[LiftableValueDependencyTreeNode]]:
        """
        Get all nodes pertinent for entity abstraction.
        :return object_nodes, case_nodes, value_nodes:
        """
        return self.find_all_objects(), self.find_all_cases(), self.find_all_values()

    def find_all_objects(self) -> List[List[LiftableObjectDependencyTreeNode]]:
        """
        Find all objects in the dependency tree.
        :return: A list containing all object enumerations in the tree.
        """
        all_object_nodes = [node for node in self.nodes() if isinstance(node, LiftableObjectDependencyTreeNode)]
        all_object_nodes.sort(key=lambda node: node.depth)
        res = []
        while len(all_object_nodes) > 0:
            node_object_list = all_object_nodes[0].get_object_list()
            if len(node_object_list) > 0:
                res.append(node_object_list)
            all_object_nodes = [node for node in all_object_nodes if node not in node_object_list]
        return res

    def find_all_cases(self) -> List[LiftableCaseDependencyTreeNode]:
        """
        Find all conditions in the dependency tree.
        :return: A list containing all conditions in the tree.
        """
        all_cases = [node for node in self.nodes() if isinstance(node, LiftableCaseDependencyTreeNode)]
        all_cases.sort(key=lambda case_node: case_node.depth)
        res = []
        for i, case in enumerate(all_cases):
            res = [result_case - case for result_case in res]
            res.append(case)
        return res

    def find_all_values(self) -> List[LiftableValueDependencyTreeNode]:
        """
        Find all values in the dependency tree.
        :return: A list containing all values in the tree.
        """
        return [node for node in self.nodes() if isinstance(node, LiftableValueDependencyTreeNode)]

    def as_conll(self) -> str:
        """
        Return the tree in the conll format.
        :return:
        """
        return "\n".join(["{0} {1} {2} {3} {4}".format(
            node.word, node.word_type,
            node.parent.node_id if node.parent != 0 else str(0),
            node.dependency, node.__class__.__name__
        ) for node in self.nodes()])

    def nodes(self) -> List[LiftableDependencyTreeNode]:
        """
        Get all nodes of the tree.
        :return:
        """
        return self.root.nodes()

    def __eq__(self, other) -> bool:
        res = False
        if isinstance(other, LiftableDependencyTree):
            res = self.root == other.root
        return res

    def __sub__(self, other: LiftableDependencyTree) -> LiftableDependencyTree:
        return LiftableDependencyTree(self.root - other.root)

    def __repr__(self):
        return f"Tree({self.root.__repr__()}, {len(self.root.nodes())})"
