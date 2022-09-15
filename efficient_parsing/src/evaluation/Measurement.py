# TODO: have a .sh script for installing the venv, all requirements, downloading using (python -m spacy download en_core_web_sm) and copying the elmo embedder for depccg_en and running the depccg en download script for elmo
from collections import namedtuple
from pathlib import Path

from typing import List, Dict, NamedTuple, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from src.candidate_reranker.configurables.reranker_configurable_enums import TableEmbedderPoolingType, \
    TableEmbeddingContent, LambdaEmbedderAttached
from src.candidate_resolver.configurables.resolver_configurable_enums import EmbeddingType, MetricLearnerType, \
    BertKerasLayerType
from src.combination.FunctionTemplate import FunctionTemplate
from src.entity_abstractor.MockAbstractor import MockAbstractor
from src.entity_abstractor.configurables.abstractor_configurable_enums import AbstractionType
from src.evaluation.PipelineTestSetCreator import PipelineTestSetCreator
from src.pipeline.SemanticParserPipeline import SemanticParserPipeline
from src.util.Storage import Storage


class Measurement:
    """
    This class defines some accuracy measurement done on a semantic parse pipeline test set.
    """
    def __init__(self,
                 measurement_name: str,
                 one_shot_generalization_test: bool = False,
                 dataset_difficulties: List[int] = None,
                 reranker_attached_configurations: List[bool] = None,
                 abstractor_configurations: List[AbstractionType] = None,
                 resolver_embedding_types: List[EmbeddingType] = None,
                 resolver_relative_not_sure_threshold: int = 2,
                 metric_learner_types: List[MetricLearnerType] = None,
                 bert_layer_types: List[BertKerasLayerType] = None,
                 table_embedder_bert_layer_types: List[BertKerasLayerType] = None,
                 table_embedder_pooling_types: List[TableEmbedderPoolingType] = None,
                 table_embedding_content_configurations: List[TableEmbeddingContent] = None,
                 reranker_lambda_embedder_attachments: List[LambdaEmbedderAttached] = None,
                 test_repetitions: int = 20):
        self.measurement_name = measurement_name
        self.one_shot_generalization_test = one_shot_generalization_test
        self.measurement_path = self.create_measurement_directories_if_necessary(measurement_name)
        self.measured_data = pd.DataFrame()
        self.false_prediction_data = pd.DataFrame()
        self.pipeline_test_inputs = self.create_pipeline_test_set(
            dataset_difficulties, reranker_attached_configurations, abstractor_configurations, resolver_embedding_types,
            resolver_relative_not_sure_threshold, metric_learner_types, bert_layer_types,
            table_embedder_bert_layer_types, table_embedder_pooling_types,
            table_embedding_content_configurations, reranker_lambda_embedder_attachments, test_repetitions
        )

    @staticmethod
    def create_pipeline_test_set(dataset_difficulties: List[int],
                                 reranker_attached_configurations: List[bool],
                                 abstractor_configurations: List[AbstractionType],
                                 resolver_embedding_types: List[EmbeddingType],
                                 resolver_relative_not_sure_threshold: int,
                                 metric_learner_types: List[MetricLearnerType],
                                 bert_layer_types: List[BertKerasLayerType],
                                 table_embedder_bert_layer_types: List[BertKerasLayerType],
                                 table_embedder_pooling_types: List[TableEmbedderPoolingType],
                                 table_embedding_content_configurations: List[TableEmbeddingContent],
                                 reranker_lambda_embedder_attachments: List[LambdaEmbedderAttached],
                                 test_repetitions: int):
        """
        Retrieve the test set of semantic parser pipelines.
        :param dataset_difficulties:
        :param reranker_attached_configurations:
        :param abstractor_configurations:
        :param resolver_embedding_types:
        :param resolver_relative_not_sure_threshold:
        :param metric_learner_types:
        :param bert_layer_types:
        :param table_embedder_bert_layer_types:
        :param table_embedder_pooling_types:
        :param table_embedding_content_configurations:
        :param reranker_lambda_embedder_attachments:
        :param test_repetitions:
        :return:
        """
        pipeline_test_set_creator = PipelineTestSetCreator(
            dataset_difficulties=dataset_difficulties,
            reranker_attached_configurations=reranker_attached_configurations,
            abstractor_configurations=abstractor_configurations,
            resolver_embedding_types=resolver_embedding_types,
            resolver_relative_not_sure_threshold=resolver_relative_not_sure_threshold,
            metric_learner_types=metric_learner_types,
            bert_layer_types=bert_layer_types,
            table_bert_layer_types=table_embedder_bert_layer_types,
            table_embedder_pooling_types=table_embedder_pooling_types,
            table_embedding_content_configurations=table_embedding_content_configurations,
            reranker_lambda_embedder_attachments=reranker_lambda_embedder_attachments,
            test_repititions=test_repetitions
        )
        return pipeline_test_set_creator.yield_pipeline_test_inputs()

    @staticmethod
    def create_measurement_directories_if_necessary(measurement_name: str):
        """
        Creates the directories where the accuracy scores and mistranslated instances are saved.
        :param measurement_name:
        :return:
        """
        measurements_directory_path = Path(__file__).parents[3] / "res" / "measurements"
        if not measurements_directory_path.is_dir():
            measurements_directory_path.mkdir()
        measurement_directory_path = measurements_directory_path / measurement_name
        if not measurement_directory_path.is_dir():
            measurement_directory_path.mkdir()
        return measurement_directory_path

    def run_tests(self):
        """
        Run the measurement for all pipeline configurations in the pipeline test set.
        :return:
        """
        i = 1
        for pipeline_test_input in self.pipeline_test_inputs:
            self.measured_data = pd.concat(
                [
                    self.measured_data,
                    pd.DataFrame([
                        self.run_atomic_action_test(*pipeline_test_input) if not self.one_shot_generalization_test
                        else self.run_one_shot_generalization_test(*pipeline_test_input)
                    ])
                ],
                ignore_index=True
            )
            self.measured_data.to_csv(
                str(self.measurement_path / f"measurment_after_{i}_tests.csv"))
            del pipeline_test_input
            i += 1
        print(self.measured_data)

    def run_atomic_action_test(self, dataset_difficulty: int, pipeline_name: str,
                               semantic_parser_pipelines: Iterable[SemanticParserPipeline]) -> Dict[str, float]:
        """
        This method runs the atomic action test feeding inputs from the dataset to the semantic parser pipeline.
        :param dataset_difficulty:
        :param pipeline_name:
        :param semantic_parser_pipelines:
        :return:
        """
        test_joined_data, test_condition_data = Storage().load_test_dataset(dataset_difficulty)
        accuracies = []
        for semantic_parser_pipeline in semantic_parser_pipelines:
            predicted_programs = []
            correct_programs = []
            for row in test_joined_data.itertuples(index=False):
                predicted_program = semantic_parser_pipeline.predict(row.query, row.context)
                condition_row = test_condition_data[test_condition_data["condition id"] == row[-1]] \
                    if row[-1] != -1 else None
                correct_program = self.retrieve_correct_program(row, condition_row)
                predicted_programs.append(predicted_program)
                correct_programs.append(correct_program)
                self.save_example_information_if_prediction_incorrect(
                    predicted_program, correct_program, row, pipeline_name
                )
            accuracies.append(self.measure_accuracy(predicted_programs, correct_programs))
        accuracies = np.array(accuracies)
        return {
            "pipeline": pipeline_name,
            "average_atomic_accuracy": accuracies.mean(),
            "worst_accuracy": np.amin(accuracies),
            "best_accuracy": np.amax(accuracies),
            "variance": np.var(accuracies)
        }

    def run_one_shot_generalization_test(self, dataset_difficulty: int, pipeline_name: str,
                                         semantic_parser_pipelines: Iterable[SemanticParserPipeline]) \
            -> Dict[str, float]:
        """
        This method implements the one-shot generalization test.
        First it gives the unknown composites to the semantic parser pipeline measureing how many have been recognized as new.
        Afterward, it retrains the semantic parser using decompositions from the dataset before checking how many compositions have been correctly generalized.
        :param dataset_difficulty:
        :param pipeline_name:
        :param semantic_parser_pipelines:
        :return:
        """
        measured_dict = {}
        composition_data, composition_condition_data, atomic_action_data = Storage().load_one_shot_generalization_data()
        for semantic_parser_pipeline in semantic_parser_pipelines:
            semantic_parser_pipeline, composition_recognition_list = self.train_programs_by_decomposition(
                semantic_parser_pipeline, composition_data, atomic_action_data
            )
            measured_dict |= self.test_new_composed_instances(
                semantic_parser_pipeline, composition_data, composition_recognition_list,
                composition_condition_data, pipeline_name
            )
        return measured_dict

    def train_programs_by_decomposition(self, semantic_parser_pipeline: SemanticParserPipeline,
                                        composition_data: pd.DataFrame,
                                        atomic_action_data: pd.DataFrame):
        """
        This method provides the semantic parser with the natural language decomposition from the compostion, and atomic action data.
        :param semantic_parser_pipeline:
        :param composition_data:
        :param atomic_action_data:
        :return:
        """
        recognized = []
        for row in composition_data.itertuples(index=False):
            prediction = semantic_parser_pipeline.predict(row[0])
            if prediction == "NOT_SURE":
                semantic_parser_pipeline = self.train_semantic_parser_pipeline_using_decomposition(
                    semantic_parser_pipeline, row, atomic_action_data
                )
                recognized.append(True)
            else:
                recognized.append(False)
        return semantic_parser_pipeline, recognized

    def test_new_composed_instances(self, semantic_parser_pipeline: SemanticParserPipeline,
                                    composition_data: pd.DataFrame, composition_recognition_list: List[bool],
                                    composition_condition_data: pd.DataFrame, pipeline_name: str):
        """
        In this method the parser checks whether the newly trained recognized composites are correctly generalized.
        :param semantic_parser_pipeline:
        :param composition_data:
        :param composition_recognition_list:
        :param composition_condition_data:
        :param pipeline_name:
        :return:
        """
        predicted_programs = []
        correct_programs = []
        for row in composition_data.itertuples():
            composition_recognized = composition_recognition_list[row[0]]
            if composition_recognized:
                condition_row = composition_condition_data[composition_condition_data["condition id"] == row[-2]] \
                    if row[-2] != -1 else None
                row_deleted = self.remove_first_row_entry(row)

                predicted_program = semantic_parser_pipeline.predict(row_deleted[0])
                correct_program = self.retrieve_correct_program(row_deleted, condition_row)
                self.save_example_information_if_prediction_incorrect(
                    predicted_program, correct_program, row_deleted, pipeline_name
                )
                predicted_programs.append(predicted_program)
                correct_programs.append(correct_program)
        return {
            "pipeline": pipeline_name,
            "recognized_instances": len([
                composition_recognized for composition_recognized in composition_recognition_list
                if composition_recognized
            ]),
            "correctly_predicted_generalizations": len([predicted_program for predicted_program, correct_program
                                                        in zip(predicted_programs, correct_programs)
                                                        if predicted_program == correct_program])
        }

    def train_semantic_parser_pipeline_using_decomposition(self, semantic_parser_pipeline: SemanticParserPipeline,
                                                           row: NamedTuple,
                                                           atomic_action_data: pd.DataFrame):
        """
        In this method the decomposition is given to the semantic parser for retraining.
        :param semantic_parser_pipeline:
        :param row:
        :param atomic_action_data:
        :return:
        """
        decomposed_atomic_action_data = self.retrieve_decomposed_atomic_action_data(
            row[3], atomic_action_data
        )
        semantic_parser_pipeline.retrain(
            row[0], decomposed_atomic_action_data["query"].tolist(),
            decomposed_atomic_action_data["context"].tolist()[0]
        )
        return semantic_parser_pipeline

    def save_example_information_if_prediction_incorrect(self, predicted_program: str, correct_program: str,
                                                         data_row: NamedTuple, pipeline_name: str):
        """
        Add mistranslated instances to a .csv file containing the pipeline configuration,
        the predicted program and the correct program as well as other information about the mistranslated instance like the difficulty.
        :param predicted_program:
        :param correct_program:
        :param data_row:
        :param pipeline_name:
        :return:
        """
        if predicted_program != correct_program:
            self.false_prediction_data = pd.concat(
                [
                    self.false_prediction_data,
                    pd.DataFrame([{
                        "pipeline_name": pipeline_name,
                        "predicted_program": predicted_program,
                        "correct_program": correct_program,
                        "query": data_row.query,
                        "difficulty": data_row.difficulty,
                        "lifted instance": data_row[1],
                        "context": data_row.context,
                        "condition_id": data_row[-1]
                    }])
                ],
                ignore_index=True)
            self.false_prediction_data.to_csv(str(self.measurement_path / "false_prediction_data.csv"))

    @staticmethod
    def retrieve_correct_program(row: NamedTuple, condition_row: pd.Series):
        """
        Retrieve the correct DSL program for the parser to check against.
        :param row:
        :param condition_row:
        :return:
        """
        abstractor = MockAbstractor()
        lifted_instance, inputs, lifted_condition = abstractor.abstract(row.query)[0]
        grounded_program = FunctionTemplate.ground_lifted_program(
            row[2], inputs, condition_row["Lifted condition DSL"].item() if condition_row is not None else None
        )
        return grounded_program

    @staticmethod
    def retrieve_decomposed_atomic_action_data(decomposition_input_ids: List[int],
                                               atomic_action_data: pd.DataFrame) -> pd.DataFrame:
        """
        Retrieve the atomic action rows in the atomic action dataset corresponding to the decomposition of a given composition example.
        :param decomposition_input_ids:
        :param atomic_action_data:
        :return:
        """
        return atomic_action_data[atomic_action_data["input id"].isin(decomposition_input_ids)]

    @staticmethod
    def measure_accuracy(predicted_programs: List[str], correct_programs: List[str]):
        accuracy = accuracy_score(correct_programs, predicted_programs, normalize=True)
        print(f"accuracy: {accuracy}")
        return accuracy

    @staticmethod
    def remove_first_row_entry(row: NamedTuple) -> NamedTuple:
        """
        Remove the first row entry of a data row from the data set.
        :param row:
        :return:
        """
        row_dict = row._asdict()
        del row_dict["Index"]
        old_keys = list(row_dict.keys())
        new_keys = [Measurement.rename_anonymous_dict_key(key) for key in row_dict.keys()]
        for i in range(len(old_keys)):
            row_dict[new_keys[i]] = row_dict.pop(old_keys[i])

        return namedtuple("DataRow", row_dict, rename=True)(**row_dict)

    @staticmethod
    def rename_anonymous_dict_key(dict_key: str):
        """
        Helper method to remove anonymous row keys from the dataset row.
        Anonymous row keys have the following form: _[0-9A-Za-z], meaning an underscore in front of a number or letter
        :param dict_key:
        :return:
        """
        if "_" == dict_key[0]:
            new_number = int(dict_key[-1]) - 1
            dict_key = dict_key.replace(dict_key[-1], str(new_number))
        return dict_key
