from src.entity_abstractor.dependencytree.nodes.LiftableDependencyTreeNode import LiftableDependencyTreeNode


class LiftableStopwordDependencyTreeNode(LiftableDependencyTreeNode):
    """
    This class represents nodes in a dependency tree which are associated with stopwords.
    """
    def __init__(self, node_id: int, word: str, word_type: str, parent_id: int, dependency: str, depth: int):
        super().__init__(node_id, word, word_type, parent_id, dependency, depth=depth)
        self.lifted_string = ""

    @classmethod
    def isinstance(cls, node: LiftableDependencyTreeNode) -> bool:
        """
        Check whether input node is a stopword node.
        :param node:
        :return:
        """
        return isinstance(node, LiftableStopwordDependencyTreeNode)

    def case_lifted(self, table=None):
        """
        Get the representation of the word corresponding to this node in the lifted condition string.
        :param table: The active table in the parser's context.
        :return:
        """
        return self.lifted(table)

    def lifted(self, table=None):
        """
        Get the representation of the word corresponding to this node in the lifted string.
        :param table: The active table in the parser's context.
        :return:
        """
        return self.lifted_string
