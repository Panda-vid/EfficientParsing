from __future__ import annotations

from smart_ccg.sem_parser.entity_abstractor.DependencyTreeLiftRule import DependencyTreeLiftRule
from smart_ccg.sem_parser.entity_abstractor.liftable_dependendency_tree.LiftableDependencyTree import LiftableDependencyTree
from smart_ccg.sem_parser.entity_abstractor.liftable_dependendency_tree.nodes.LiftableDependencyTreeNode import LiftableDependencyTreeNode


class DependencyTreeLifter:
    def __init__(self, rules):
        self.rules = rules

    @classmethod
    def get_instance(cls) -> DependencyTreeLifter:
        pass

    def lift(self, tree: LiftableDependencyTree) -> None:
        for node in tree.nodes():
            applicable_rule = self.find_rule_by_node(node)
            applicable_rule.apply(node)

    def find_rule_by_node(self, node: LiftableDependencyTreeNode) -> DependencyTreeLiftRule:
        applicable_rules = [rule for rule in self.rules if rule.is_applicable(node)]
        if len(applicable_rules) > 1:
            raise RuntimeError("The lifting rule set is ambiguous!")
        elif len(applicable_rules) == 0:
            result = DependencyTreeLiftRule.empty_rule()
        else:
            result = applicable_rules[0]
        return result

