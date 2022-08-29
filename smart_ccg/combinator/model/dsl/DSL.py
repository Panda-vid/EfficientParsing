from typing import List

from smart_ccg.combinator.model.dsl.dsl_entities.ProtoAction import ProtoAction


class DSL:
    def __init__(self, actions: List[ProtoAction], subactions: List[ProtoAction], conjunctions: List[str],
                 condition: str, conditionals: List[str]):
        self.actions = actions
        self.subactions = subactions
        self.conjunctions = conjunctions
        self.condition = condition
        self.conditionals = conditionals

    def get_actions_of_type(self, action_name):
        return [action for action in self.actions if action.type()[1] == action_name]

    def get_subactions_of_type(self, action_name):
        return [action for action in self.subactions if action.type()[1] == action_name]