import random

from typing import List, Tuple

import pandas as pd


from smart_ccg.combinator.model.dsl import DSL
from smart_ccg.combinator.model.dsl.dsl_entities.Action import Action
from smart_ccg.combinator.model.dsl.dsl_entities.Assignment import Assignment
from smart_ccg.combinator.model.dsl.dsl_entities.Condition import Condition
from smart_ccg.combinator.model.dsl.dsl_entities.DSLList import DSLList
from smart_ccg.combinator.model.dsl.dsl_entities.Input import Input
from smart_ccg.combinator.model.dsl.dsl_entities.ProtoAction import ProtoAction
from smart_ccg.combinator.model.dsl.dsl_entities.SimpleInput import SimpleInput
from smart_ccg.combinator.model.dsl.dsl_entities.Type import Type


class Combinator:

    def __init__(self, dsl: DSL, tables: List[pd.DataFrame]):
        self.dsl = dsl
        self.tables = tables

    def generate_randomized_example(self, depth: int, maxlistdepth: int = 5) -> Tuple[str, pd.DataFrame]:
        selected_table = random.choice(self.tables)
        actions = []
        conjunctions = []
        for i in range(depth):
            selected_proto_action = random.choice(self.dsl.actions)
            action_inputs = self.get_input_for_proto_action(selected_proto_action, selected_table,
                                                            maxlistdepth)
            if selected_proto_action.is_valid(action_inputs):
                actions.append(Action(selected_proto_action, action_inputs))
            else:
                raise RuntimeError("Cannot create Action from selected ProtoAction and Inputs")
            if i + 1 <= depth:
                conjunctions.append(random.choice(self.dsl.conjunctions))
        example_output = generate_example_output(actions, conjunctions)
        return example_output, selected_table

    def get_input_for_proto_action(self, proto_action: ProtoAction, table: pd.DataFrame,
                                   maxlistdepth: int) -> List[Input]:
        action_inputs = []
        for input_type in proto_action.input_types:
            action_inputs.append(self.get_input_from_input_type(input_type, table, maxlistdepth))
        return action_inputs

    def get_input_from_input_type(self, input_type: Type, table: pd.DataFrame, maxlistdepth: int) -> Input:
        if type(input_type) is tuple:
            res_input = self.get_randomized_complex_input(input_type, table, maxlistdepth)
        elif input_type == Type.CONDITION:
            res_input = self.get_randomized_condition(table)
        # or it is a SimpleInput of Type.COLUMN, Type.VALUE or Type.TABLE
        else:
            res_input = self.get_simple_input(input_type, table)

        return res_input

    # TODO input generation can be part of the corresponding classes to reduce clutter
    def get_randomized_complex_input(self, input_type: Type, table: pd.DataFrame, maxlistdepth: int) -> Input:
        res_input = None
        if input_type[0] == Type.LIST:
            res_input = self.get_list_input(input_type[1], table, maxlistdepth)
        elif input_type[0] == Type.ASSIGNMENT:
            assignee_type, value_type = input_type[1]
            res_input = self.get_randomized_assignment(assignee_type, value_type, table)
        elif input_type[0] == Type.ACTION:
            res_input = self.get_randomized_action(input_type, table, maxlistdepth)

        return res_input

    def get_randomized_assignment(self, assignee_type: Type, value_type: Type, table: pd.DataFrame) -> Assignment:
        assignee = self.get_simple_input(assignee_type, table)
        value = self.get_simple_input(value_type, table)
        return Assignment(assignee, value)

    def get_randomized_condition(self, table: pd.DataFrame) -> Condition:
        # TODO: or ask user for condition
        conditional = random.choice(self.dsl.conditionals)
        selected_column = random.choice(table.columns)
        condition_value = random.choice(table[selected_column])
        return Condition(self.dsl, conditional, selected_column, condition_value)

    def get_randomized_action(self, input_type: Type, table: pd.DataFrame, maxlistdepth: int) -> Action:
        fitting_dsl_subactions = self.dsl.get_subactions_of_type(input_type[1])
        fitting_dsl_actions = self.dsl.get_actions_of_type(input_type[1])
        selected_action = random.choice(fitting_dsl_actions + fitting_dsl_subactions)
        action_inputs = self.get_input_for_proto_action(selected_action, table, maxlistdepth)
        return selected_action.create_action(action_inputs)

    def get_simple_input(self, input_type: Type, table: pd.DataFrame) -> SimpleInput:
        res_input = None
        if input_type == Type.VALUE:
            # TODO: ask user for value.
            pass
        elif input_type == Type.COLUMN:
            content = random.choice(table.columns)
            res_input = SimpleInput(input_type, content)
        elif input_type == Type.TABLE:
            content = "this_table"
            res_input = SimpleInput(input_type, content)
        return res_input

    def get_list_input(self, content_input_type: Type, table: pd.DataFrame, maxlistdepth: int) -> DSLList:
        inputs = []
        for i in range(maxlistdepth):
            inputs.append(self.get_input_from_input_type(content_input_type, table, maxlistdepth))
        return DSLList(inputs)


def generate_example_output(actions: List[Action], conjunctions: List[str]) -> str:
    flattened = []
    for i in range(len(actions)):
        flattened.append(actions[i].dsl_output())
        if i < len(conjunctions):
            flattened.append(conjunctions[i])

    return " ".join(flattened)
