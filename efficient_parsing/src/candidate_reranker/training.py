import tensorflow as tf
import pandas as pd

from typing import List

from src.candidate_reranker.lambda_calculus_embedding.LambdaEmbedder import LambdaEmbedder
from src.candidate_reranker.table_embedding.TableEmbedder import TableEmbedder
from src.datamodel.Table import Table


def create_reranker_training_data_from(reranker_dataframe: pd.DataFrame,
                                       tables: List[Table],
                                       lambda_embedder: LambdaEmbedder,
                                       table_embedder: TableEmbedder):
    features = []
    labels = []
    for table_name in reranker_dataframe["table_name"].unique():
        table = list(filter(lambda candidate_table: candidate_table.table_name == table_name, tables))[0]
        sub_dataframe = reranker_dataframe[reranker_dataframe["table_name"] == table_name]
        for row in sub_dataframe.itertuples(index=False):
            feature = create_reranker_input(row.query, table, lambda_embedder, table_embedder)
            positive_feature = tf.concat([tf.constant([[1.0]], dtype=tf.float32), feature], axis=-1)
            negative_feature = tf.concat([tf.constant([[0.0]], dtype=tf.float32), feature], axis=-1)
            features = append_reranker_feature(positive_feature, features)
            features = append_reranker_feature(negative_feature, features)
            labels.append(1)
            labels.append(0)
    return features, tf.transpose(tf.constant([labels], dtype=tf.int32))


def create_reranker_input(query: str,
                          table: Table,
                          lambda_embedder: LambdaEmbedder,
                          table_embedder: TableEmbedder):
    embedded_lambda_calc = lambda_embedder(query)
    embedded_table = table_embedder.embedd(table, query)
    return tf.concat([
            tf.constant(embedded_lambda_calc, dtype=tf.float32)[tf.newaxis, :], embedded_table
        ], -1)


def append_reranker_feature(feature: tf.Tensor, features: tf.Tensor):
    if len(features) == 0:
        features = feature
    else:
        features = tf.concat([features, feature], axis=0)
    return features
