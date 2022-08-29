import itertools
import random

import tensorflow as tf
import numpy as np
import pandas as pd

from pathlib import Path

from src.candidate_resolver.CandidateResolver import CandidateResolver
from src.candidate_resolver.Regressor import Regressor
from src.candidate_resolver.scorers import euclidean_distance


def create_and_train_model(
        dataset: pd.DataFrame,
        lifted_instance_column_name: str,
        lifted_program_column_name: str,
        model_location: Path,
        embedding_function,
        self_attention: bool = False,
        metric_learner=None,
        regressor_num_epochs: int = 1,
        regressor_learning_rate: float = 1e-5,
        take_best_guess: bool = False):
    embedded_dataset = embedd_dataset(
        dataset, lifted_instance_column_name, lifted_program_column_name, embedding_function
    )
    regressor = Regressor()
    regressor.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=regressor_learning_rate),
                      loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=[tf.keras.metrics.BinaryCrossentropy()])
    model = CandidateResolver(embedding_function,
                              regressor,
                              model_location,
                              self_attention=self_attention,
                              nearest_neighbor_metric_learner=metric_learner,
                              take_best_guess=take_best_guess)
    max_positive_distance = train_model(model, embedded_dataset, 0, 1, regressor_num_epochs)
    model.not_sure_threshold = max_positive_distance / 4
    model.save()
    return model


def embedd_dataset(dataset: pd.DataFrame,
                   lifted_instance_column_name: str,
                   lifted_program_column_name: str,
                   embedding_function) -> pd.DataFrame:
    embedded_dataset = pd.DataFrame()
    embedded_dataset[lifted_instance_column_name] = dataset[lifted_instance_column_name].apply(embedding_function)
    embedded_dataset[lifted_program_column_name] = dataset[lifted_program_column_name]
    return embedded_dataset


def train_model(
        model: CandidateResolver,
        embedded_dataset: pd.DataFrame,
        feature_column_index: int,
        label_column_index: int,
        regressor_num_epochs: int = 1,
        equal_positive_negative: bool = False):
    model.train_instances_from(embedded_dataset, feature_column_index, label_column_index)
    if equal_positive_negative:
        x, y, max_positive_distance = equal_positive_and_negative_regressor_training_pairs_from(model)
    else:
        positive_train, negative_train, max_positive_distance = get_all_regressor_training_pairs_from(model)
        train = positive_train + negative_train
        train = np.array(train)
        x = train[:, 0]
        y = train[:, 1]
    model.regressor.fit(x=tf.expand_dims(x, axis=-1), y=tf.expand_dims(y, axis=-1), epochs=regressor_num_epochs,
                        verbose=1, validation_split=0.0)
    return max_positive_distance


def equal_positive_and_negative_regressor_training_pairs_from(model: CandidateResolver):
    positive_examples, negative_examples, max_positive_distance = get_all_regressor_training_pairs_from(model)
    len_positive_examples = len(positive_examples)
    len_negative_examples = len(negative_examples)
    if len_positive_examples >= len_negative_examples:
        positive_examples = random.sample(positive_examples, len_negative_examples)
    else:
        negative_examples = random.sample(negative_examples, len_positive_examples)
    result = positive_examples + negative_examples
    random.shuffle(result)
    result = np.array(result)
    return result[:, 0], result[:, 1], max_positive_distance


def get_all_regressor_training_pairs_from(model: CandidateResolver):
    positive_train_examples = []
    negative_train_examples = []
    max_positive_distance = 0
    for i, j in itertools.combinations(range(model.trained_set_size() - 1), 2):
        dsl_output_i, feature_vector_i = model.trained_instance_at(i)
        dsl_output_j, feature_vector_j = model.trained_instance_at(j)
        if i != j:
            if dsl_output_j == dsl_output_i:
                distance = euclidean_distance(feature_vector_i, feature_vector_j)
                positive_train_examples.append([
                    distance,
                    1.0
                ])
                if distance > max_positive_distance:
                    max_positive_distance =  distance
            else:
                negative_train_examples.append([
                    euclidean_distance(feature_vector_i, feature_vector_j),
                    0.0
                ])
    return positive_train_examples, negative_train_examples, max_positive_distance

