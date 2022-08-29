from typing import Tuple, List

from src.candidate_resolver.configurables.resolver_configurable_enums import EmbeddingType
from src.candidate_resolver.embedding.BertEmbedder import BertEmbedder
from src.candidate_resolver.embedding.embedding_utils import summed_quadratic_kernel
from src.candidate_resolver.embedding.postprocessing.postprocessors import CustomPooling, PoolingType, PaddingType


class EmbeddingFunctionProvider:

    def __init__(self,
                 embedder: BertEmbedder,
                 pooling_window_shape: Tuple = (128, 32),
                 number_of_strides: Tuple = (1, 32),
                 pooling_padding_type: PaddingType = PaddingType.VALID):
        self.embedder = embedder
        self.pooling_arguments = [pooling_window_shape, number_of_strides, pooling_padding_type]
        self.pooling_window_shape = pooling_window_shape
        self.number_of_strides = number_of_strides
        self.pooling_padding_type = pooling_padding_type

    def select_embedding_function(self,
                                  embedding_type: EmbeddingType,
                                  pooling_window_shape: Tuple = None,
                                  number_of_strides: Tuple = None,
                                  padding_type: PaddingType = None):
        match embedding_type:
            case EmbeddingType.SEQUENCE:
                return self.get_sequence_embedding_function()
            case EmbeddingType.BERT_POOLED:
                return self.get_bert_pooled_embedding_function()
            case EmbeddingType.SEQUENCE_POSITIONAL:
                return self.get_sequence_positional_encoded_embedding_function()
            case EmbeddingType.MAX_POOLED:
                return self.get_max_pooled_embedding_function(
                    *self.resolve_pooling_arguments([
                        pooling_window_shape,
                        number_of_strides,
                        padding_type
                    ])
                )
            case EmbeddingType.AVG_POOLED:
                return self.get_avg_pooled_embedding_function(
                    *self.resolve_pooling_arguments([
                        pooling_window_shape,
                        number_of_strides,
                        padding_type
                    ])
                )
            case EmbeddingType.MAX_POOLED_POSITIONAL:
                return self.get_max_pooled_positional_encoded_embedding_function(
                    *self.resolve_pooling_arguments([
                        pooling_window_shape,
                        number_of_strides,
                        padding_type
                    ])
                )
            case EmbeddingType.AVG_POOLED_POSITIONAL:
                return self.get_avg_pooled_positional_encoded_embedding_function(
                    *self.resolve_pooling_arguments([
                        pooling_window_shape,
                        number_of_strides,
                        padding_type
                    ])
                )
            case EmbeddingType.SUMMED_QUADRATIC_KERNEL_SEQUENCE:
                return self.get_summed_quadratic_kernel_of_sequence_embedding_function()
            case EmbeddingType.SUMMED_QUADRATIC_KERNEL_SEQUENCE_POSITIONAL:
                return self.get_summed_quadratic_kernel_of_positional_encoded_sequence_embedding_function()
            case EmbeddingType.SUMMED_QUADRATIC_KERNEL_BERT_POOLED:
                return self.get_summed_quadratic_kernel_of_bert_pooled_embedding_function()
            case _:
                # noinspection PyUnreachableCode
                raise NotImplementedError(f"No function supplier for {embedding_type} exists!")

    def get_summed_quadratic_kernel_of_sequence_embedding_function(self):
        return self.get_summed_quadratic_kernel_function(
            self.get_sequence_embedding_function()
        )

    def get_summed_quadratic_kernel_of_positional_encoded_sequence_embedding_function(self):
        return self.get_summed_quadratic_kernel_function(
            self.get_sequence_positional_encoded_embedding_function()
        )

    def get_summed_quadratic_kernel_of_bert_pooled_embedding_function(self):
        return self.get_summed_quadratic_kernel_function(
            self.get_bert_pooled_embedding_function()
        )

    def get_max_pooled_positional_encoded_embedding_function(
            self,
            pooling_window_shape,
            number_of_strides,
            padding_type):
        return self.get_pooled_embedding_function(
            self.get_sequence_positional_encoded_embedding_function(),
            PoolingType.MAX,
            pooling_window_shape,
            number_of_strides,
            padding_type
        )

    def get_avg_pooled_positional_encoded_embedding_function(
            self,
            pooling_window_shape: Tuple,
            number_of_strides: Tuple,
            padding_type: PaddingType):
        return self.get_pooled_embedding_function(
            self.get_sequence_positional_encoded_embedding_function(),
            PoolingType.AVG,
            pooling_window_shape,
            number_of_strides,
            padding_type
        )

    def get_max_pooled_embedding_function(self,
                                          pooling_window_shape,
                                          number_of_strides,
                                          padding_type):
        return self.get_pooled_embedding_function(
            self.get_sequence_embedding_function(),
            PoolingType.MAX,
            pooling_window_shape,
            number_of_strides,
            padding_type
        )

    def get_avg_pooled_embedding_function(self,
                                          pooling_window_shape: Tuple,
                                          number_of_strides: Tuple,
                                          padding_type: PaddingType):
        return self.get_pooled_embedding_function(
            self.get_sequence_embedding_function(),
            PoolingType.AVG,
            pooling_window_shape,
            number_of_strides,
            padding_type
        )

    @staticmethod
    def get_pooled_embedding_function(
            sequence_embedding_function,
            pooling_type: PoolingType,
            pooling_window_shape: Tuple,
            number_of_strides: Tuple,
            padding_type: PaddingType):
        def func(inputs):
            sequence_embedder = sequence_embedding_function
            pooling = CustomPooling(
                pooling_window_shape,
                number_of_strides,
                pooling_type,
                padding_type
            )
            return pooling(sequence_embedder(inputs))
        return func

    @staticmethod
    def get_summed_quadratic_kernel_function(embedding_function):
        return lambda inputs: summed_quadratic_kernel(embedding_function(inputs), embedding_function(inputs))

    def get_bert_pooled_embedding_function(self):
        return lambda inputs: self.embedder(inputs)["pooled_output"]

    def get_sequence_embedding_function(self):
        return lambda inputs: self.embedder(inputs)["sequence_output"]

    def get_sequence_positional_encoded_embedding_function(self):
        return lambda inputs: self.embedder(inputs)["sequence_output_pos_encoded"]

    def resolve_pooling_arguments(self, pooling_arguments: List):
        return (
            arg if arg is not None else self.pooling_arguments[i]
            for i, arg in enumerate(pooling_arguments)
        )

