import pickle

import tensorflow as tf
import numpy as np
import pandas as pd

from pathlib import Path

from src.candidate_resolver.NearestNeighbors import NearestNeighbors


class CandidateResolver:
    def __init__(self,
                 embedding_function,
                 regressor: tf.keras.Model,
                 save_location: Path,
                 take_best_guess: bool = False,
                 nearest_neighbor_metric_learner=None,
                 not_sure_threshold: float = 400,
                 self_attention: bool = False):
        super().__init__()
        self.embedding_function = embedding_function
        self.regressor = regressor
        self.self_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=2, key_dim=2, attention_axes=(1, 2)
        ) if self_attention else None
        self.not_sure_threshold = not_sure_threshold
        self.take_best_guess = take_best_guess
        self.trained_examples = []
        self.transformed_trained_examples = np.array([[]])
        self.nearest_neighbor_search = None
        self.nearest_neighbor_metric_learner = nearest_neighbor_metric_learner
        self.dsl_programs = []
        self.save_location = Path(save_location)

    @staticmethod
    def load(save_location: Path, embedding_function):
        with open(save_location / Path("model_attributes.pkl"), "rb") as file:
            model = pickle.load(file)
        model.regressor = tf.keras.models.load_model(save_location / "regressor")
        model.embedding_function = embedding_function
        return model

    def call(self, inputs, training=False):
        inputs = self.embedding_function(inputs)
        inputs = inputs[tf.newaxis, :]
        inputs = inputs if self.nearest_neighbor_metric_learner is None \
            else self.nearest_neighbor_metric_learner.transform(inputs[0])[tf.newaxis, :]
        inputs = inputs if self.self_attention is None else self.self_attention(inputs, inputs, training=training)
        distances, candidate_programs = self.close_enough_examples(np.array(inputs))
        output_probabilities = [
            tf.squeeze(self.regressor(np.array(distance)[np.newaxis, np.newaxis]))
            for distance in distances
        ]
        return (output_probabilities, candidate_programs) \
            if not self.take_best_guess or len(output_probabilities) == 0 \
            else (output_probabilities[0], candidate_programs[0])

    def add_training_example_and_retrain(self, lifted_instance: str, dsl_program: str):
        self.trained_examples = np.append(self.trained_examples, self.embedding_function(lifted_instance), axis=0)
        self.dsl_programs.append(dsl_program)
        self.train_metric_if_required()
        self.create_nearest_neighbor_classifier()

    def close_enough_examples(self, inputs):
        example_indices, distances = self.nearest_neighbor_search.query_radius(inputs)
        candidate_programs = [self.dsl_programs[i] for i in example_indices]
        return distances, candidate_programs

    def train_instances_from(self,
                             embedded_dataframe: pd.DataFrame,
                             feature_column_index: int,
                             label_column_index: int):
        self.extract_feature_vectors_and_dsl_programs_from(
            embedded_dataframe, feature_column_index, label_column_index
        )
        self.create_nearest_neighbor_classifier()

    def extract_feature_vectors_and_dsl_programs_from(self,
                                                      embedded_dataframe: pd.DataFrame,
                                                      feature_column_index: int,
                                                      label_column_index: int):
        self.add_feature_vectors_and_dsl_programs_from(embedded_dataframe, feature_column_index, label_column_index)
        self.train_metric_if_required()

    def add_feature_vectors_and_dsl_programs_from(self,
                                                  embedded_dataframe: pd.DataFrame,
                                                  feature_column_index: int,
                                                  label_column_index: int):
        for row in embedded_dataframe.itertuples(index=False):
            row_feature_vector = np.array(row[feature_column_index])
            row_dsl_output = row[label_column_index]
            self.add_trained_feature_vector(row_feature_vector)
            self.dsl_programs.append(row_dsl_output)

    def train_metric_if_required(self):
        if self.nearest_neighbor_metric_learner is not None:
            self.transformed_trained_examples = self.nearest_neighbor_metric_learner.fit_transform(
                self.trained_examples,
                self.dsl_programs
            )

    def create_nearest_neighbor_classifier(self):
        self.nearest_neighbor_search = NearestNeighbors(
            self.trained_examples if self.nearest_neighbor_metric_learner is None
            else self.transformed_trained_examples,
            self.not_sure_threshold,
        )

    def add_trained_feature_vector(self, feature_vector: np.ndarray):
        self.trained_examples = feature_vector if len(self.trained_examples) == 0 \
            else np.vstack((self.trained_examples, feature_vector))

    def trained_instance_at(self, i: int):
        return (self.dsl_programs[i], self.trained_examples[i, :]) if self.nearest_neighbor_metric_learner is None \
            else (self.dsl_programs[i], self.transformed_trained_examples[i, :])

    def trained_set_size(self):
        return len(self.dsl_programs)

    def save(self):
        self.regressor.save(self.save_location / Path("regressor"))
        with open(self.save_location / Path("model_attributes.pkl"), "wb") as file:
            pickle.dump(self, file)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["embedding_function"]
        del state["regressor"]
        return state
