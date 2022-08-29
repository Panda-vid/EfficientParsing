import numpy as np
import pandas as pd
import tensorflow as tf

from typing import List, NamedTuple

from src.candidate_reranker.RerankerNetwork import RerankerNetwork
from src.candidate_reranker.lambda_calculus_embedding.LambdaEmbedder import LambdaEmbedder
from src.candidate_reranker.table_embedding.TableEmbedder import TableEmbedder
from src.candidate_resolver.CandidateResolver import CandidateResolver
from src.datamodel.Table import Table
from src.util.Storage import Storage


class CandidateReranker:
    def __init__(self,
                 table_embedder: TableEmbedder,
                 lambda_embedder: LambdaEmbedder,
                 reranker_data: pd.DataFrame):
        self.lambda_embedder = lambda_embedder
        self.table_embedder = table_embedder
        self.reranker_data = reranker_data
        self.reranker_network = None
        self.reranker_features = []
        self.reranker_labels = []

    @classmethod
    def create_and_train(cls,
                         candidate_resolver: CandidateResolver,
                         reranker_data: pd.DataFrame,
                         table_embedder: TableEmbedder,
                         reranker_network_dropout_hidden_layer: float = 0.4,
                         reranker_network_dropout_output_layer: float = 0.4,
                         reranker_network_num_epochs: int = 2,
                         reranker_network_learning_rate: float = 1e-3,
                         lambda_embedder: LambdaEmbedder = None):
        candidate_reranker = cls(table_embedder, lambda_embedder, reranker_data)
        feature_data_output_dimension = \
            candidate_reranker.generate_reranker_training_data(reranker_data, candidate_resolver)
        candidate_reranker.reranker_network = RerankerNetwork.create_and_train(
            candidate_reranker.reranker_features,
            tf.convert_to_tensor(candidate_reranker.reranker_labels, dtype=tf.int32),
            (None, feature_data_output_dimension), int(feature_data_output_dimension/2),
            1, reranker_network_dropout_hidden_layer,
            reranker_network_dropout_output_layer, reranker_network_num_epochs,
            reranker_network_learning_rate
        )
        return candidate_reranker

    def select_best_candidate(self,
                              utterance: str,
                              resolver_probabilities: tf.Tensor,
                              candidate_programs: List[str],
                              tables: List[Table]) -> str:
        embedded = self.embedd_data(utterance, tables)
        reranker_input = self.prepend_model_output_probabilities(embedded, resolver_probabilities)
        reranker_probabilities = self.reranker_network.predict(reranker_input).flatten()
        best_candidate_index = np.argmax(reranker_probabilities)
        return candidate_programs[best_candidate_index]

    def retrain(self, utterance: str, tables: List[Table], lifted_candidate_program: str):
        self.append_data_entry(utterance, tables)
        if self.lambda_embedder is not None:
            self.lambda_embedder.retrain(lifted_candidate_program)
        dropout_hidden_layer, dropout_output_layer, network_num_epochs, network_learning_rate = \
            self.reranker_network.get_hyperparams()
        self.create_and_train(self.reranker_data, self.table_embedder, dropout_hidden_layer, dropout_output_layer,
                              network_num_epochs, network_learning_rate, self.lambda_embedder)

    def generate_reranker_training_data(self, reranker_data: pd.DataFrame, candidate_resolver: CandidateResolver):
        reranker_feature_dim = 0
        for row in reranker_data.itertuples(index=False):
            self.generate_examples_from_row_and_add_to_model(row, candidate_resolver)
        return reranker_feature_dim

    def generate_examples_from_row_and_add_to_model(self, row: NamedTuple, candidate_resolver: CandidateResolver):
        storage = Storage()
        utterance = row.query
        matching_tables = storage.get_matching_tables(row.table)
        query_table_aligned_embedding = self.embedd_data(utterance, matching_tables)
        lifted_instance, true_lifted_program = row[1], row[2]
        self.generate_examples_from_candidate_resolver_and_add_to_model(
            candidate_resolver, query_table_aligned_embedding, lifted_instance, true_lifted_program
        )

    def generate_examples_from_candidate_resolver_and_add_to_model(self, candidate_resolver: CandidateResolver,
                                                                   query_table_aligned_embedding: tf.Tensor,
                                                                   lifted_instance: str, true_lifted_program: str):
        resolver_output_probabilities, candidate_programs = candidate_resolver.call(lifted_instance)
        for candidate_program, output_probability in zip(candidate_programs, resolver_output_probabilities):
            feature_vector, correct_probability = self.generate_training_example_from(
                candidate_program, true_lifted_program, output_probability, query_table_aligned_embedding
            )
            self.append_reranker_training_example(feature_vector, correct_probability)

    def generate_training_example_from(self, candidate_program: str, true_program: str,
                                       resolver_output_probability: tf.Tensor,
                                       query_table_aligned_embedding: tf.Tensor):
        reranker_input_feature = tf.concat([
            resolver_output_probability[tf.newaxis, tf.newaxis],
            query_table_aligned_embedding
        ], axis=-1)
        return (reranker_input_feature, 0) if candidate_program != true_program else (reranker_input_feature, 1)

    def append_data_entry(self, utterance: str, tables: List[Table]):
        self.reranker_data = self.reranker_data.append(
           {
               "query": utterance,
               "table": [[table.table_name for table in tables]]}
        )

    def embedd_data(self, utterance: str, tables: List[Table]) -> tf.Tensor:
        embedded_lambda_calc = self.generate_lambda_embedding_if_required(utterance)
        embedded_tables = self.generate_summed_table_embedding(tables, utterance)
        return self.concatenate_embeddings_if_required(embedded_lambda_calc, embedded_tables)

    def append_reranker_training_example(self, feature: tf.Tensor, label: int):
        self.reranker_labels.append(label)
        if len(self.reranker_features) == 0:
            self.reranker_features = feature
        else:
            self.reranker_features = tf.concat([self.reranker_features, feature], axis=0)

    @staticmethod
    def prepend_model_output_probabilities(embedded_data: tf.Tensor, output_probabilities):
        embedded_data = tf.tile(embedded_data, [output_probabilities.shape[0], 1])
        return tf.keras.layers.concatenate([output_probabilities, embedded_data], axis=1)

    def generate_lambda_embedding_if_required(self, utterance: str):
        return self.lambda_embedder(utterance) if self.lambda_embedder is not None else None

    def generate_summed_table_embedding(self, tables: List[Table], utterance: str):
        summed_table_embedding = self.table_embedder.embedd(tables[0], utterance).numpy()
        for table in tables[1:]:
            summed_table_embedding += self.table_embedder.embedd(table, utterance).numpy()
        return tf.convert_to_tensor(summed_table_embedding, np.float32)

    def concatenate_embeddings_if_required(self,
                                           embedded_lambda_calc: tf.Tensor, embedded_table: tf.Tensor) -> tf.Tensor:
        return tf.concat([
            tf.constant(embedded_lambda_calc, dtype=tf.float32)[tf.newaxis, :], embedded_table
        ], -1) if self.lambda_embedder is not None else embedded_table
