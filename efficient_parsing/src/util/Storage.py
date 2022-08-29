import ast
from functools import reduce
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from src.datamodel.Table import Table


class Storage:
    def __init__(self):
        resources_location = Path(__file__).parents[3] / "res"
        self.dataset_location = resources_location / "dataset"

    def load_candidate_resolver_dataset(self, difficulty: int = 3) -> pd.DataFrame:
        main_data = self.load_main_data(self.training_data_location)
        main_data = main_data[main_data["difficulty"] <= difficulty]
        return main_data[["Lifted instance", "DSL output"]]

    def load_reranking_dataset(self, difficulty: int = 3) -> pd.DataFrame:
        main_data = self.load_main_data(self.training_data_location)
        input_data = self.load_input_data(self.training_data_location)
        joined_data = main_data.merge(input_data, on="input id")
        joined_data = joined_data[joined_data["difficulty"] <= difficulty]
        return joined_data[["query", "Lifted instance", "DSL output", "context", "table"]]

    def load_abstraction_recombination_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        joined_test_data, test_condition_data = self.load_test_dataset()
        main_data = self.load_main_data(self.training_data_location)
        input_data = self.load_input_data(self.training_data_location)
        condition_data = self.load_condition_input_data(self.training_data_location)
        joined_input_data = input_data.merge(main_data, on="input id")
        joined_input_data = pd.concat([joined_input_data, joined_test_data], ignore_index=True)
        condition_data = pd.concat([condition_data, test_condition_data], ignore_index=True)
        return (
            joined_input_data[["query", "Lifted instance", "DSL output", "column", "table", "context", "condition id"]],
            condition_data
        )

    def load_test_dataset(self, difficulty: int = 3) -> Tuple[pd.DataFrame, pd.DataFrame]:
        main_data = self.load_main_data(self.test_data_location)
        input_data = self.load_input_data(self.test_data_location)
        condition_data = self.load_condition_input_data(self.test_data_location)
        joined_data = main_data.merge(input_data, on="input id")
        joined_data = joined_data[joined_data["difficulty"] <= difficulty]
        return (
            joined_data[
                ["query", "Lifted instance", "DSL output", "column", "table", "context", "difficulty", "condition id"]
            ],
            condition_data
        )

    def load_one_shot_generalization_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        composition_data = self.load_composition_data()
        composition_input_data = self.load_input_data(self.one_shot_data_location)
        composition_condition_data = self.load_condition_input_data(self.one_shot_data_location)
        atomic_data = self.load_main_data(self.training_data_location)
        atomic_input_data = self.load_input_data(self.training_data_location)

        joined_composition_data = composition_data.merge(composition_input_data, on="input id")
        joined_atomic_data = atomic_data.merge(atomic_input_data, on="input id")
        joined_composition_data.rename({
            "Composite query": "query",
            "Lifted composite instance": "Lifted instance",
            "DSL": "DSL output"
        }, axis="columns", inplace=True)
        composition_data_length = len(joined_composition_data.index)
        joined_composition_data["difficulty"] = np.repeat(3, composition_data_length)
        return joined_composition_data, composition_condition_data, joined_atomic_data

    def load_composition_abstraction_recombination_data(self):
        composition_data = self.load_composition_data()
        composition_input_data = self.load_input_data(self.one_shot_data_location)
        composition_condition_data = self.load_condition_input_data(self.one_shot_data_location)
        joined_composition_data = composition_data.merge(composition_input_data, on="input id")
        joined_composition_data.rename({
            "Composite query": "query",
            "Lifted composite instance": "Lifted instance",
            "DSL": "DSL output"
        }, axis="columns", inplace=True)
        composition_data_length = len(joined_composition_data.index)
        joined_composition_data["difficulty"] = np.repeat(3, composition_data_length)
        return joined_composition_data, composition_condition_data

    def get_matching_tables(self, table_names: List[List[str]]) -> List[Table]:
        return [
            table for table in self.load_all_tables()
            if table.table_name in reduce(lambda pack1, pack2: pack1 + pack2, table_names)
        ]

    def load_all_tables(self) -> List[Table]:
        all_tables = []
        for table_context in self.table_contexts_location.iterdir():
            if table_context.is_dir():
                all_tables += self.load_context(table_context.stem)
        return all_tables

    def load_context(self, context: str) -> List[Table]:
        context_folder = self.table_contexts_location / context
        return [
            Table(pd.read_csv(str(table_path), sep="\t"), table_path.stem, context_folder.name)
            for table_path in context_folder.iterdir()
            if table_path.is_file() and table_path.suffix == ".txt"
        ]

    def load_table(self, context: str, table_name: str) -> Table:
        table_path = (self.table_contexts_location / context).with_stem(table_name)
        if table_path.is_file():
            return Table(pd.read_csv(str(table_path), sep="\t"), table_name, context)
        else:
            raise FileNotFoundError(f"Path: {str(table_path)} is no file.")

    @property
    def table_contexts_location(self):
        return self.dataset_location / "tables"

    @property
    def training_data_location(self):
        return self.dataset_location / "train"

    @property
    def test_data_location(self):
        return self.dataset_location / "test"

    @property
    def one_shot_data_location(self):
        return self.dataset_location / "composition"

    def load_composition_data(self):
        return pd.read_csv(
            str(self.one_shot_data_location / "compositions.txt"), sep="\t",
            dtype={
                "Composite query": str, "lifted composite instance": str, "input id": int
            },
            converters={
                "decomposed query input ids": ast.literal_eval
            },
            index_col=False
        ).fillna("-")

    @staticmethod
    def load_main_data(dataset_folder: Path) -> pd.DataFrame:
        return pd.read_csv(
            str(dataset_folder / "data.txt"), sep="\t",
            dtype={
                "query": str, "Lifted instance": str, "DSL output": str, "input id": int, "difficulty": int
            },
            index_col=False
        ).fillna("-")

    @staticmethod
    def load_input_data(dataset_folder: Path) -> pd.DataFrame:
        return pd.read_csv(
            str(dataset_folder / "inputs.txt"), sep="\t",
            dtype={
                "input id": int, "context": str, "condition id": int
            },
            converters={
                "table": ast.literal_eval,
                "column": ast.literal_eval
            },
            index_col=False
        ).fillna("-")

    @staticmethod
    def load_condition_input_data(dataset_folder: Path) -> pd.DataFrame:
        return pd.read_csv(
            str(dataset_folder / "conditions.txt"), sep="\t",
            dtype={
                "condition id": int, "condition DSL": str, "Lifted condition": str,
                "Lifted condition DSL": str
            },
            converters={
                "condition column": ast.literal_eval, "condition value": ast.literal_eval
            },
            index_col=False
        ).fillna("-")
