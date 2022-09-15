from __future__ import annotations

import copy
from typing import List, Dict, Any, Tuple

from src.datamodel.Table import Table
from src.datamodel.TableType import TableType
from src.entity_abstractor.dependencytree.LiftableDependencyTree import LiftableDependencyTree
from src.entity_abstractor.dependencytree.nodes.LiftableCaseDependencyTreeNode import LiftableCaseDependencyTreeNode
from src.entity_abstractor.dependencytree.nodes.LiftableObjectDependencyTreeNode import LiftableObjectDependencyTreeNode
from src.entity_abstractor.dependencytree.nodes.LiftableValueDependencyTreeNode import LiftableValueDependencyTreeNode
from src.entity_abstractor.utils import normalized_compounds, compounds


class Sentence:
    """
    This class defines the internal Sentence representation of the abstraction algorithm.
    Each sentence can be abstracted and contains a dependency tree which is defined by the subclassses of LiftableDependencTreeNode and the LiftableDependencyTree class.
    """
    def __init__(self, dependency_tree: LiftableDependencyTree,
                 objects: List[List[LiftableObjectDependencyTreeNode]],
                 cases: List[LiftableCaseDependencyTreeNode],
                 values: List[LiftableValueDependencyTreeNode]):
        """
        The constructor of the Sentence class.
        :param dependency_tree: The liftable dependency tree corresponding to a given input utterance.
        :param objects: All nodes in the dependency tree corresponding to objects.
        :param cases: All nodes in the dependency tree corresponding to sentence cases.
        :param values: All nodes in the dependency tree corresponding to values in a sentence.
        """
        self.dependency_tree = dependency_tree
        self.objects = objects
        self.cases = cases
        self.values = values

    @classmethod
    def create_sentence(cls, dependency_tree: LiftableDependencyTree) -> Sentence:
        """
        Creates a Sentence instance from a given dependency tree.
        :param dependency_tree: The liftable dependency tree corresponding to a given input utterance.
        :return sentence: The sentence instance corresponding to the input utterance
        """
        objects, cases, values = dependency_tree.find_all_important_nodes()
        return cls(dependency_tree, objects, cases, values)

    def abstract(self, table: Table) -> Tuple[str, Dict[str, Any], str]:
        """
        Abstract the sentence.
        :param table: The active table in the parser's context.
        :return lifted_sentence_string, input_dict, lifted_condition:
        """
        return self.lifted(table), self.get_input_dict(table), self.case_lifted(table)

    def lifted(self, table: Table = None) -> str:
        """
        Get the lifted string from the sentence.
        :param table: The active table in the parser's context.
        :return:
        """
        nonempty_lifted_strings = [
            node.lifted(table) for node in self.dependency_tree.nodes() if node.lifted(table) != ""
        ]
        return " ".join(nonempty_lifted_strings)

    def case_lifted(self, table: Table = None) -> str:
        """
        Get the lifted condition from the sentence.
        :param table: The active table in the parser's context.
        :return:
        """
        res = []
        for case in self.cases:
            case_nodes = [
                node.case_lifted(table) for node in case.nodes() if node.case_lifted(table) != ""
            ]
            if len(case_nodes) > 1:
                res.append(" ".join(case_nodes))
        return res[0] if len(res) > 0 else None

    def get_input_dict(self, table: Table) -> Dict[str, Any]:
        """
        Get all inputs extracted from the sentence in a dict where the keys are the DSL data types and the values are their respective values.
        :param table: The active table in the parser's context.
        :return:
        """
        res = {}
        column_names = self.get_lifted_column_names(table)
        table_names = self.get_lifted_table_names(table)
        cases = self.get_case_input_dicts(table)
        if len(column_names) > 0:
            res[TableType.COLUMN.value] = column_names
        if len(table_names) > 0:
            res[TableType.TABLE.value] = table_names
        if len(cases) > 0:
            res[TableType.CONDITION.value] = cases
        return res

    def get_lifted_column_names(self, table: Table) -> List[List[str]]:
        """
        Get all column names lifted from the sentence.
        :param table: The active table in the parser's context.
        :return:
        """
        column_names = []
        for obj_pack in self.objects:
            column_name_pack = self.get_column_name_pack(obj_pack, table)
            if len(column_name_pack) > 0:
                column_names.append(column_name_pack)
        return column_names

    def get_lifted_table_names(self, table: Table) -> List[List[str]]:
        """
        Get all table names lifted from the sentence.
        :param table: The active table in the parser's context.
        :return:
        """
        res = []
        for obj_pack in self.objects:
            table_pack = self.get_table_name_pack(obj_pack, table)
            if len(table_pack) > 0:
                res.append(table_pack)

        return res if len(res) > 0 else ([[table.table_name]] if table is not None else [[]])

    def get_case_input_dicts(self, table: Table) -> List[Dict[str, str]]:
        """
        Analogous to the input dict for sentences this creates an input dict for the conditions found in the sentence.
        :param table: The active table in the parser's context.
        :return:
        """
        column_names = self.get_case_lifted_column_names(table)
        values = self.get_lifted_values()
        input_dicts = []
        for column_name, value in zip(column_names, values):
            input_dicts.append(
                {
                    TableType.COLUMN.value: column_name,
                    TableType.VALUE.value: value
                }
            )
        return input_dicts

    def get_case_lifted_column_names(self, table: Table) -> List[List[str]]:
        """
        Finds the column names inside the conditions of the sentence.
        :param table: The active table in the parser's context.
        :return:
        """
        column_names = []
        for obj_pack in self.objects:
            for obj in obj_pack:
                if obj.get_table_type(table) == TableType.COLUMN and obj.is_oblique_predecessor_of_case():
                    for normalized_compound in normalized_compounds(obj):
                        if table.is_column_name(normalized_compound) and normalized_compound not in column_names:
                            column_names.append(normalized_compound)
        return column_names

    def get_lifted_values(self) -> List[str]:
        """
        Finds the values inside the conditions of the sentence.
        :return:
        """
        all_values = copy.deepcopy(self.values)
        all_values.sort(key=lambda value_node: value_node.depth)
        res = []
        for case in self.cases:
            value = [
                neighbor for neighbor in case.get_neighbors()
                if LiftableValueDependencyTreeNode.isinstance(neighbor)
            ]
            value = case.get_lifted_value_node_from_children() if len(value) == 0 else value[0]
            if value is not None:
                try:
                    all_values.index(value)
                    all_values.remove(value)
                    res.append(value.word)
                except ValueError:
                    continue
        return res

    @staticmethod
    def get_column_name_pack(obj_pack: List[LiftableObjectDependencyTreeNode], table: Table) -> List[str]:
        """
        Gets the column names in a list for each discrete enumeration of columns in the sentence.
        :param obj_pack: An enumeration of objects .
        :param table: The active table in the parser's context.
        :return:
        """
        column_name_pack = []
        for obj in obj_pack:
            if obj.get_table_type(table) == TableType.COLUMN and not obj.is_oblique_predecessor_of_case():
                if table is not None:
                    column_name_pack = Sentence.find_correct_normalized_compound(obj, table, column_name_pack)
                else:
                    column_name_pack.append(normalized_compounds(obj)[0].replace("column_", ""))
        return column_name_pack

    @staticmethod
    def get_table_name_pack(obj_pack: List[LiftableObjectDependencyTreeNode], table: Table) -> List[str]:
        """
        Gets the table names in a list for each discrete enumeration of tables in the sentence.
        :param obj_pack: An enumeration of objects .
        :param table: The active table in the parser's context.
        :return:
        """
        table_name_pack = []
        for obj in obj_pack:
            if obj.get_table_type(table) == TableType.TABLE:
                if table is not None:
                    table_name_pack = Sentence.find_correct_compound(obj, table, table_name_pack)
                else:
                    table_name_pack.append(compounds(obj)[0].replace("table_", ""))
        return table_name_pack

    @staticmethod
    def find_correct_normalized_compound(obj: LiftableObjectDependencyTreeNode, table: Table,
                                         column_name_pack: List[str]) -> List[str]:
        """
        This method combines normalized word compounds into compounds according to different camel and snake case naming conventions.
        Normalized word compounds are compounds consisting of only singular word stems.
        :param obj: The object node associated with the compound
        :param table: The active table in the parser's context.
        :param column_name_pack: An enumeration of column names.
        :return:
        """
        for normalized_compound in normalized_compounds(obj):
            if table.is_column_name(normalized_compound) and normalized_compound not in column_name_pack:
                column_name_pack.append(normalized_compound)
        return column_name_pack

    @staticmethod
    def find_correct_compound(obj: LiftableObjectDependencyTreeNode, table: Table,
                              table_name_pack: List[str]) -> List[str]:
        """
        This method combines word compounds into compounds according to different camel and snake case naming conventions.
        :param obj: The object node associated with the compound
        :param table: The active table in the parser's context.
        :param table_name_pack: An enumeration of table names.
        :return:
        """
        for compound in compounds(obj):
            if table.is_table_name(compound):
                table_name_pack.append(compound)
        return table_name_pack
