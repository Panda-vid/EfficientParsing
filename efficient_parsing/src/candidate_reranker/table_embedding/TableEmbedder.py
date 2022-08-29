import math
from typing import List, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_text as text
import tensorflow_hub as hub
import gensim.downloader as api

from sklearn.metrics.pairwise import cosine_similarity

from src.candidate_reranker.table_embedding.utils import join
from src.datamodel.Table import Table
from src.util.string_utils import remove_punctuation
from src.util.tfhub_utils import delete_matching_tfhub_cache


class TableEmbedder:
    def __init__(self,
                 tfhub_preprocessor_link: str,
                 tfhub_bert_link: str,
                 gensim_api_word_embedder: str,
                 sequence_length: int,
                 column_names_only: bool,
                 pooling_layer_window_shape,
                 pooling_output_shape,
                 pooling_type: str):
        preprocessor = hub.load(tfhub_preprocessor_link)
        special_tokens_dict = preprocessor.tokenize.get_special_tokens_dict()
        self.sep_token = tf.expand_dims(special_tokens_dict["start_of_sequence_id"], axis=0)
        self.cls_token = tf.expand_dims(special_tokens_dict["end_of_segment_id"], axis=0)
        self.tokenizer = hub.KerasLayer(preprocessor.tokenize)
        self.sequence_length = sequence_length
        self.bert_layer = hub.KerasLayer(tfhub_bert_link, trainable=False)
        if not column_names_only:
            self.word_embedding_model = api.load(gensim_api_word_embedder)
        self.column_names_only = column_names_only
        self.pooling_layer_window_shape = pooling_layer_window_shape
        self.pooling_output_shape = pooling_output_shape
        self.pooling_type = pooling_type

    @classmethod
    def initialize(cls,
                   tfhub_preprocessor_link: str,
                   tfhub_bert_link: str,
                   gensim_api_word_embedder: str = "glove-wiki-gigaword-200",
                   sequence_length: int = 128,
                   column_names_only: bool = False,
                   pooling_layer_window_shape=(128, 128),
                   pooling_output_shape=(1, 27),
                   pooling_type: str = "MAX"):
        try:
            embedder = cls(
                tfhub_preprocessor_link,
                tfhub_bert_link,
                gensim_api_word_embedder,
                sequence_length,
                column_names_only,
                pooling_layer_window_shape,
                pooling_output_shape,
                pooling_type
            )
        except tf.errors.DataLossError as error:
            print("Data loss found: redownloading model.")
            delete_matching_tfhub_cache(error)
            embedder = cls.initialize(
                tfhub_preprocessor_link,
                tfhub_bert_link,
                gensim_api_word_embedder,
                sequence_length,
                column_names_only,
                pooling_layer_window_shape,
                pooling_output_shape,
                pooling_type
            )
        return embedder

    def embedd(self, table: Table, query: str) -> tf.Tensor:
        return self.create_feature_from_column_names(table, query) \
            if self.column_names_only \
            else self.create_feature_from_data(table, query)

    def create_feature_from_column_names(self, table: Table, query: str) -> tf.Tensor:
        columns = [" ".join(table.columns)]
        return self.create_feature(table, query, columns)

    def create_feature_from_data(self, table: Table, query: str) -> tf.Tensor:
        entries = [" ".join(entry) for entry in self.get_linearized_data_entries(table)]
        entries = self.get_sorted_entries(entries, query)
        return self.create_feature(table, query, entries)

    @staticmethod
    def get_linearized_data_entries(table: Table):
        columns = table.columns
        for entry in table.get_entries():
            for col_index, col_value in enumerate(entry):
                yield f"{columns[col_index]} {col_value}"

    def get_sorted_entries(self, table_entries: List[str], query: str) -> List[str]:
        query_average_wordvector = self.get_average_wordvector_of(query)
        distances = np.array([
            cosine_similarity(query_average_wordvector.reshape(1, -1),
                              self.get_average_wordvector_of(entry).reshape(1, -1))
            for entry in table_entries
        ]).flatten()
        indices = (-distances).argsort()
        return [table_entries[i] for i in indices]

    def create_feature(self,
                       table: Table,
                       query: str,
                       table_data_to_be_embedded: List[str]
                       ) -> tf.Tensor:
        tokenized_features = self.get_tokenized_features(query, table.get_contexts(), table_data_to_be_embedded)
        combined = tf.concat([self.cls_token[:, tf.newaxis, tf.newaxis],
                              join(tokenized_features, self.sep_token[:, tf.newaxis, tf.newaxis]),
                              self.sep_token[:, tf.newaxis, tf.newaxis]], 1)
        data, mask = text.pad_model_inputs(input=combined, max_seq_length=self.sequence_length)
        segment_mask = self.get_segment_mask(data.numpy().flatten())
        bert_input = {"input_word_ids": data, "input_mask": mask, "input_type_ids": segment_mask[tf.newaxis, :]}
        bert_output = self.bert_layer(bert_input)["sequence_output"][:, :, :, tf.newaxis]
        return tf.nn.pool(
            bert_output,
            self.pooling_layer_window_shape,
            self.pooling_type,
            strides=self.get_strides(bert_output.shape[1:3])
        )[0, :, :, 0]

    def get_tokenized_features(self, query: str, contexts: List[str], table_entries: List[str]):
        tokenized_features = [self.tokenize(query)] \
                             + [self.tokenize(context) for context in contexts] \
                             + [self.tokenize(entry) for entry in table_entries]
        return self.select_tokenized_features(tokenized_features)

    def get_segment_mask(self, padded_data: np.ndarray) -> tf.Tensor:
        separator_indices = np.where(padded_data.flatten() == self.sep_token)[0]
        first_separator_index = separator_indices[0]
        last_separator_index = separator_indices[-1]
        first_segment_mask = np.zeros(first_separator_index + 1)
        second_segment_mask = np.ones(last_separator_index - first_separator_index)
        padding_segment_mask = np.zeros(len(padded_data) - (last_separator_index + 1))
        return tf.constant(np.hstack((first_segment_mask, second_segment_mask, padding_segment_mask)),
                           dtype=tf.int32)

    def get_strides(self, bert_output_shape) -> Tuple[int, int]:
        return (
            int(
                math.ceil(float(bert_output_shape[0]) - self.pooling_layer_window_shape[0] + 1)/
                self.pooling_output_shape[0]
            ), int(
                math.ceil(float(bert_output_shape[1]) - self.pooling_layer_window_shape[1] + 1) /
                self.pooling_output_shape[1]
            )
        )

    def get_average_wordvector_of(self, sequence: str) -> np.ndarray:
        return np.array([
            self.word_embedding_model[word]
            for word in list(map(remove_punctuation, sequence.lower().split()))
            if self.word_embedding_model.has_index_for(word)
        ]).mean(axis=0)

    def tokenize(self, inputs: str):
        inputs = tf.constant([inputs], dtype=tf.string)
        return self.tokenizer(inputs)

    def select_tokenized_features(self, tokenized_features: List[tf.Tensor]) -> List[tf.Tensor]:
        number_of_tokens = 1
        for i, tokenized_feature in enumerate(tokenized_features):
            number_of_tokens += len(tokenized_feature.numpy().flatten()) + 1
            if number_of_tokens > self.sequence_length:
                return tokenized_features[:i]
        return tokenized_features
