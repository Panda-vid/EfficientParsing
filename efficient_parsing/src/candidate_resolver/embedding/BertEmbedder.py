import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

from typing import Dict
from efficient_parsing.src.candidate_resolver.embedding.embedding_utils import apply_positional_encoding
from src.util.tfhub_utils import delete_matching_tfhub_cache


class BertEmbedder(tf.keras.Model):
    def __init__(self, tfhub_preprocessor_link: str, tfhub_bert_link: str):
        super().__init__()
        self.text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
        self.preprocess = hub.KerasLayer(tfhub_preprocessor_link, trainable=False)
        self.bert_layer = hub.KerasLayer(tfhub_bert_link, trainable=False)

    @classmethod
    def initialize(cls, tfhub_preprocessor_link: str, tfhub_bert_link: str):
        try:
            embedder = cls(tfhub_preprocessor_link, tfhub_bert_link)
        except tf.errors.DataLossError as error:
            print("Data loss found: redownloading model.")
            delete_matching_tfhub_cache(error)
            embedder = cls.initialize(tfhub_preprocessor_link, tfhub_bert_link)
        return embedder

    def call(self, inputs: str) -> Dict[str, tf.Tensor]:
        inputs = tf.constant([inputs], dtype=tf.string)
        bert_inputs = self.preprocess.call(inputs)
        bert_outputs = self.bert_layer.call(bert_inputs)
        sequence_output_with_positional_encoding = apply_positional_encoding(bert_outputs["sequence_output"],
                                                                             bert_inputs["input_mask"])
        bert_outputs["sequence_output_pos_encoded"] = sequence_output_with_positional_encoding
        return bert_outputs
