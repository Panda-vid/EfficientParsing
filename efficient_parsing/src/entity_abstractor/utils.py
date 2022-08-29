from typing import List


from src.entity_abstractor.dependencytree.nodes.LiftableCompoundDependencyTreeNode import \
    LiftableCompoundDependencyTreeNode
from src.entity_abstractor.dependencytree.nodes.LiftableObjectDependencyTreeNode import LiftableObjectDependencyTreeNode
from src.util.string_utils import normalize, make_snake_case, make_camel_case, make_capital_camel_case, make_compound, \
    make_space_separated_compound, make_capital_snake_case, make_all_capital_snake_case, make_upper_snake_case


def compounds(object_node: LiftableObjectDependencyTreeNode):
    return create_variable_compound_versions(compound_words(object_node, normalized=False))


def normalized_compounds(object_node: LiftableObjectDependencyTreeNode):
    return create_variable_compound_versions(compound_words(object_node))


def compound_words(object_node: LiftableObjectDependencyTreeNode, normalized: bool = True) -> List[str]:
    initial_word = normalize(object_node.word) if normalized else object_node.word
    object_node.children.sort(key=lambda node: node.node_id)
    res = [initial_word]
    for child in object_node.children:
        child_word = normalize(child.word) if normalized else child.word
        initial_word_index = res.index(initial_word)
        if LiftableCompoundDependencyTreeNode.isinstance(child):
            if child.node_id > object_node.node_id:
                res.append(child_word)
            else:
                res.insert(initial_word_index, child_word)
    return res


def create_variable_compound_versions(words: List[str]) -> List[str]:
    return [
        make_snake_case(words),
        make_capital_snake_case(words),
        make_all_capital_snake_case(words),
        make_upper_snake_case(words),
        make_camel_case(words),
        make_capital_camel_case(words),
        make_compound(words),
        make_space_separated_compound(words)
    ]
