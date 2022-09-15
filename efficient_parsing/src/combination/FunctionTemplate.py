import re

from enum import Enum
from typing import Tuple, Dict, Any, Iterable, List

from src.datamodel.TableType import TableType


class Multiplicity(Enum):
    NONE = 0
    SINGLE = 1
    MULTIPLE = 2


class FunctionTemplate:
    """
    Grounds program templates by filling in the inputs extracted by the abstractor.
    (e.g., SELECT([table], [,column]) -> SELECT(['time_slots'], ['date']))
    """
    def __init__(self, lifted_program: str, lifted_condition: str = None):
        self.assigment = self.extract_assignment(lifted_program)
        self.standard_fill_tuples = self.extract_standard_fill_tuples_from(self.assigment)
        self.condition_assignment = self.extract_assignment(lifted_condition) if lifted_condition is not None else {}
        self.condition_standard_fill_tuple = self.extract_standard_fill_tuples_from(self.condition_assignment)
        self.output_function_template = self.create_output_function_template_from(lifted_program, self.assigment)
        if self.condition_assignment != {}:
            self.condition_template = self.create_output_function_template_from(
                lifted_condition, self.condition_assignment
            )

    @classmethod
    def ground_lifted_program(cls, lifted_program: str, inputs: Dict[str, Any], lifted_condition: str = None) -> str:
        function_template = cls(lifted_program, lifted_condition)
        return function_template.resolve_input(inputs)

    @staticmethod
    def extract_assignment(output_function_template: str) -> Dict[Tuple[str, int], Tuple[List[int], Multiplicity]]:
        """
        Get the function signature of a given template by finding all DSL type placeholders and creating a dict describing the signature.
        The signature dict contains keys with the DSL type and a number describing which input of the type it is, if there are multiple input lists of the same type.
        The values are the position inside the function template format string which gets filled and the multiplicity of the input.
        (e.g., {(column, 0): (0, single), (table, 0): (1, single), (column, 1): (2, multiple)})
        :param output_function_template:
        :return assignment:
        """
        assignment = {}
        for position, input_parameter_information \
                in enumerate(FunctionTemplate.extract_input_signature(output_function_template)):
            table_type, number, multiplicity = input_parameter_information
            if (table_type, number) not in assignment.keys():
                assignment[(table_type, number)] = ([position], multiplicity)
            else:
                assignment[(table_type, number)][0].append(position)
        return assignment

    @staticmethod
    def extract_standard_fill_tuples_from(assignment: Dict[Tuple[str, int], Tuple[List[int], Multiplicity]]) \
            -> Tuple[int, str]:
        """
        This enables partial grounding by introducing another assignment dict which corresponds to filling the placeholders inside the program template format string.
        :param assignment: The function template input signature.
        :return:
        """
        standard_fill_tuples = []
        for (table_type, number), (position, multiplicity) in assignment.items():
            multiplicity_token = "" if multiplicity == Multiplicity.SINGLE else ","
            standard_fill_tuples.append(
                (position, f"[{multiplicity_token}{table_type}{number if number != 0 else ''}]")
            )
        standard_fill_tuples.sort(key=lambda fill_tuple: fill_tuple[0])
        return standard_fill_tuples

    @staticmethod
    def create_output_function_template_from(lifted_functions: str,
                                             assignment: Dict[Tuple[str, int], Tuple[List[int], Multiplicity]])\
            -> str:
        """
        This method creates a format string for the given function template.
        (e.g. "SELECT([table], [,column])" -> "SELECT({0}, {1})")
        :param lifted_functions:
        :param assignment: The function template input signature.
        :return:
        """
        for key, value in assignment.items():
            table_type, type_pack_number = key
            template_positions, multiplicity = value
            program = re.compile(r"\[(,)?" + f"({table_type})" + r"(\d)?]")
            for template_position in template_positions:
                lifted_functions = program.sub(
                    "{%s}" % template_position,
                    lifted_functions,
                    count=1
                )
        return lifted_functions

    @staticmethod
    def extract_input_signature(output_function_template: str) -> Iterable[Tuple[str, int, Multiplicity]]:
        """
        Find all DSL type tags and extract the input signature for a single match of the DSL type tag inside the function template.
        :param output_function_template:
        :return:
        """
        for match in re.finditer(r"\[(,)?(.+?)(\d)?]", output_function_template):
            yield FunctionTemplate.extract_input_parameter_information_from(match)

    @staticmethod
    def extract_input_parameter_information_from(match: re.Match) -> Tuple[str, int, Multiplicity]:
        """
        Extract the input signature for a single match of the DSL type tag inside the function template.
        If a comma is found inside the tag, the multiplicity is 'multiple' else its single.
        The remaining text inside the tag corresponds to a DSL input type.
        :param match:
        :return:
        """
        multiplicity = Multiplicity.MULTIPLE if match.group(1) == "," else Multiplicity.SINGLE
        table_type = match.group(2)
        number = int(match.group(3)) if match.group(3) is not None else 0
        return table_type, number, multiplicity

    def resolve_input(self, inputs: Dict[str, Any]):
        """
        Fill the format string with the inputs from the entity abstractor.
        If input types are found which cannot be found inside the function signature return an empty string.
        :param inputs:
        :return:
        """
        function_template_fill_tuples = []
        for table_type, values in inputs.items():
            for pack_number, value_pack in enumerate(values):
                if self.is_valid_function_input(table_type, pack_number, value_pack):
                    template_fill_tuple = self.get_assignment_template_fill_tuple(table_type, pack_number, value_pack)
                    function_template_fill_tuples += template_fill_tuple
                else:
                    return ""
        function_template_fill_tuples.sort(key=lambda fill_tuple: fill_tuple[0])
        ordered_format_input = [fill_tuple[1] for fill_tuple in function_template_fill_tuples]
        input_format_positions = [fill_tuple[0] for fill_tuple in function_template_fill_tuples]

        for positions, standard_input in self.standard_fill_tuples:
            for position in positions:
                if position not in input_format_positions:
                    input_format_positions.append(position)
                    input_format_positions.sort()
                    index = input_format_positions.index(position)
                    ordered_format_input.insert(index, standard_input)

        format_input_tuple = tuple(ordered_format_input)
        return self.output_function_template.format(*format_input_tuple)

    def get_assignment_template_fill_tuple(self,
                                           input_table_type: str,
                                           input_pack_number: int,
                                           value_pack: List[str]) -> Tuple[int, str]:
        """
        Creates an input signature for a condition the same way it is being done for the function template.
        :param input_table_type:
        :param input_pack_number:
        :param value_pack:
        :return:
        """
        positions = self.get_positions_in_template_of_input(input_table_type, input_pack_number)
        return [
            (
                position,
                (
                    str(value_pack)
                    if input_table_type != TableType.CONDITION.value
                    else self.get_grounded_condition(value_pack)
                )
            ) for position in positions
        ]

    def get_grounded_condition(self, value_pack: Dict[str, str] | List[Dict[str, str]]) -> Tuple[int, str]:
        """
        Ground a condition template. The implementation is analogous to the implementation for function templates.
        :param value_pack:
        :return:
        """
        condition_template_fill_tuples = []
        if isinstance(value_pack, list):
            for condition_pair_number, condition_input in enumerate(value_pack):
                value_position = self.get_position_in_condition_template(
                    TableType.VALUE.value, condition_pair_number
                )
                column_position = self.get_position_in_condition_template(
                    TableType.COLUMN.value, condition_pair_number
                )
                value_tuple = (value_position, condition_input[TableType.VALUE])
                column_tuple = (column_position, condition_input[TableType.COLUMN])
                condition_template_fill_tuples.append(value_tuple)
                condition_template_fill_tuples.append(column_tuple)
        else:
            condition_template_fill_tuples = [
                (self.condition_assignment[table_type, 0][0], input_value)
                for table_type, input_value in value_pack.items()
            ]
        condition_template_fill_tuples.sort(key=lambda fill_tuple: fill_tuple[0])
        ordered_format_input_tuple = (fill_tuple[1] for fill_tuple in condition_template_fill_tuples)
        return self.condition_template.format(*ordered_format_input_tuple)

    def is_valid_function_input(self,
                                input_table_type: str,
                                input_pack_number: int,
                                input_value_pack: List[str] | Dict[str, str]):
        return (input_table_type, input_pack_number) in self.assigment.keys() \
               and (self.has_correct_multiplicity(input_table_type, input_pack_number, input_value_pack)
                    or input_table_type == TableType.CONDITION.value)

    def has_correct_multiplicity(self,
                                 input_table_type: str,
                                 input_pack_number: int,
                                 input_value_pack: List[str]):
        multiplicity = self.get_required_multiplicity_for_input_with(input_table_type, input_pack_number)
        return ((len(input_value_pack) == 1 and multiplicity == Multiplicity.SINGLE)
                or (len(input_value_pack) >= 1 and multiplicity == Multiplicity.MULTIPLE))

    def get_positions_in_template_of_input(self, input_table_type: str, input_pack_number: int) -> int:
        return self.assigment[input_table_type, input_pack_number][0]

    def get_position_in_condition_template(self, condition_table_type: str, condition_pair_number: int) -> int:
        return self.condition_assignment[condition_table_type, condition_pair_number][0][0]

    def get_required_multiplicity_for_input_with(self, input_table_type: str, input_pack_number: int) -> Multiplicity:
        return self.assigment[input_table_type, input_pack_number][1]
