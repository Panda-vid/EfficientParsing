import unittest

from src.combination.FunctionTemplate import FunctionTemplate
from src.datamodel.TableType import TableType
from src.entity_abstractor.MockAbstractor import MockAbstractor


class CombinationTest(unittest.TestCase):

    def test_function_grounding(self):
        mock_abstractor_out = MockAbstractor.abstract_utterance("Show the name, surname, email and phone.")
        grounded_program = FunctionTemplate.ground_lifted_program(
            "SELECT([table], [,column])", mock_abstractor_out[0][1]
        )
        self.assertEqual("SELECT(['time_slots'], ['name', 'surname', 'email', 'phone'])", grounded_program)

    def test_combined_grounding(self):
        mock_abstractor_out = MockAbstractor.abstract_utterance("Count all rows where date is later than today.")
        grounded_program = FunctionTemplate.ground_lifted_program(
            "FILTER([table], [condition]); COUNT([table])", mock_abstractor_out[0][1], "[column] <= [value]"
        )
        self.assertEqual("FILTER(['time_slots'], date <= today); COUNT(['time_slots'])", grounded_program)

    def test_partial_grounding(self):
        partial_inputs = {TableType.COLUMN.value: [['name', 'address']]}
        partially_grounded_program = FunctionTemplate.ground_lifted_program(
            "SELECT([table], [,column])", partial_inputs
        )
        self.assertEqual("SELECT([table], ['name', 'address'])", partially_grounded_program)

    def test_inputs_disagree(self):
        mock_abstractor_out = MockAbstractor.abstract_utterance("Count all rows where date is later than today.")
        grounded_program = FunctionTemplate.ground_lifted_program(
            "SELECT([table], [,column])", mock_abstractor_out[0][1]
        )
        self.assertEqual("", grounded_program)
