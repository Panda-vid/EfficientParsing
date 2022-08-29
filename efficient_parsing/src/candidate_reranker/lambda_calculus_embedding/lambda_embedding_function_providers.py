import re
import spacy

import numpy as np
import pandas as pd

from typing import Callable, Tuple, List


nlp = spacy.load("en_core_web_sm")


def create_lambda_embedder_function(dataset: pd.DataFrame,
                                    label_column_name: str,
                                    no_hit_included: bool = False) -> Callable[[str], np.ndarray]:
    lemma_one_hot_embedder = create_one_hot_embedder(
        *create_label_mapping(dataset, label_column_name)
    )

    def embedding_output_function(lambda_calc: str):
        embedded_vector = np.sum(
            np.array([
                lemma_one_hot_embedder(lemma) for lemma in extract_lemmas_from(lambda_calc)
            ]), 0)
        return embedded_vector if no_hit_included else embedded_vector[:-1]

    return embedding_output_function


def create_one_hot_embedder(mapping: dict, vec_length: int) -> Callable[[str], np.ndarray]:
    return lambda word: create_one_hot_vector(vec_length, mapping[word]) \
        if word in mapping.keys() \
        else create_one_hot_vector(vec_length, vec_length - 1)


def create_label_mapping(
        dataframe: pd.DataFrame,
        label_column_name: str) -> Tuple[dict, int]:
    classes = list(dataframe[label_column_name].unique())
    return {
               lemmatized_word: classes.index(entry[2])
               for entry in dataframe.itertuples(index=False)
               for lemmatized_word in get_lemmatized_words_in(entry[1])
           }, len(classes) + 1


def extract_lemmas_from(lambda_calc: str) -> List[str]:
    return [match[1] for match in re.findall("(_)([a-z]+)", lambda_calc)]


def create_one_hot_vector(length: int, hot_index: int) -> np.ndarray:
    return np.eye(1, length, k=hot_index).flatten()


def get_lemmatized_words_in(lifted_instance: str):
    document = nlp(filter_lifted(lifted_instance))
    return [token.lemma_ for token in document if token.pos_]


def filter_lifted(lifted_instance: str):
    return re.sub(r"(\[[A-Za-z,]+\])", "", lifted_instance).strip()
