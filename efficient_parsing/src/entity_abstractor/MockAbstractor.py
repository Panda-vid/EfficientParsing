from typing import Dict, Any, List, Tuple

import pandas as pd

from src.datamodel.Table import Table
from src.datamodel.TableType import TableType

from src.util.Storage import Storage


class MockAbstractor:
    def __init__(self):
        storage = Storage()
        self.input_data, self.condition_data = storage.load_abstraction_recombination_data()
        self.composition_input_data, self.composition_condition_data = \
            storage.load_composition_abstraction_recombination_data()

    @classmethod
    def abstract_utterance(cls, utterance: str) -> List[Tuple[str, Dict[str, Any], str]]:
        abstractor = MockAbstractor()
        return abstractor.abstract(utterance)

    def abstract(self, utterance: str, table: Table = None) -> List[Tuple[str, Dict[str, Any], str]]:
        filtered_data_set_example = self.input_data[self.input_data["query"] == utterance]
        if len(filtered_data_set_example.index) != 0:
            lifted_condition, condition_inputs = self.get_condition_input_if_possible(filtered_data_set_example)
        else:
            filtered_data_set_example = self.composition_input_data[self.composition_input_data["query"] == utterance]
            lifted_condition, condition_inputs = \
                self.get_composition_condition_input_if_possible(filtered_data_set_example)
        lifted_instance = filtered_data_set_example["Lifted instance"].values[0]
        column_input = filtered_data_set_example["column"].values[0] \
            if filtered_data_set_example["column"].values[0] != [[]] else []
        table_input = filtered_data_set_example["table"].values[0]
        inputs = self.pack_inputs(column_input, table_input, condition_inputs)
        return [(lifted_instance, inputs, lifted_condition)]

    def get_condition_input_if_possible(self, filtered_data_set_example: pd.Series) \
            -> Tuple[str, List[Tuple[str, str]]]:
        if filtered_data_set_example["condition id"].values[0] != -1:
            filtered_data_set_example = filtered_data_set_example.merge(
                self.condition_data[
                    self.condition_data["condition id"] == filtered_data_set_example["condition id"].values[0]
                ]
            )
            return self.extract_condition_data(filtered_data_set_example)
        return None, None

    def get_composition_condition_input_if_possible(self, filtered_data_set_example: pd.Series):
        if filtered_data_set_example["condition id"].values[0] != -1:
            filtered_data_set_example = filtered_data_set_example.merge(
                self.composition_condition_data[
                    self.composition_condition_data["condition id"] ==
                    filtered_data_set_example["condition id"].values[0]
                ]
            )
            return self.extract_condition_data(filtered_data_set_example)
        return None, None

    @staticmethod
    def extract_condition_data(filtered_data_set_example: pd.Series):
        lifted_condition = filtered_data_set_example["Lifted condition"].values[0]
        column_condition_inputs = filtered_data_set_example["condition column"].values[0]
        value_condition_inputs = filtered_data_set_example["condition value"].values[0]
        condition_inputs = [
            {
                TableType.COLUMN.value: column_condition_input,
                TableType.VALUE.value: value_condition_input
            } for column_condition_input, value_condition_input
            in zip(column_condition_inputs, value_condition_inputs)
        ]
        return lifted_condition, condition_inputs

    @staticmethod
    def pack_inputs(column_input: List[List[str]],
                    table_input: List[List[str]],
                    condition_input: List[Dict[str, str]]) -> Dict[str, Any]:
        res = {}
        if len(column_input) > 0:
            res[TableType.COLUMN.value] = column_input
        if len(table_input) > 0:
            res[TableType.TABLE.value] = table_input
        if condition_input is not None:
            res[TableType.CONDITION.value] = condition_input
        return res
