from smart_ccg.sem_parser.entity_abstractor.DependencyTreeLiftRule import DependencyTreeLiftRule
from smart_ccg.sem_parser.entity_abstractor.liftable_dependendency_tree.nodes.LiftableDependencyTreeNode import LiftableDependencyTreeNode

RULES = [
    object_list_rule(),
    case_rule(),
    case_list_rule(),
    value_rule(),
    comparison_rule()
]

def object_list_rule() -> DependencyTreeLiftRule:
    parent_dependency = LiftableDependencyTreeNode.OBJECT_DEPENDENCY
    node_dependency = LiftableDependencyTreeNode.CONJUNCTION_DEPENDENCY

def case_rule() -> DependencyTreeLiftRule:
    pass

def value_rule() -> DependencyTreeLiftRule:
    pass

def comparison_rule() -> DependencyTreeLiftRule:
    pass

def resolve_neighboring_object_nodes()