import tensorflow as tf

from enum import Enum
from typing import Tuple


class PoolingType(Enum):
    MAX = "MAX"
    AVG = "AVG"


class PaddingType(Enum):
    SAME = "SAME"
    VALID = "VALID"


class CustomPooling:
    def __init__(self, pooling_window_shape: Tuple,
                 number_of_strides: Tuple,
                 pooling_type: PoolingType,
                 padding_type: PaddingType):

        self.pooling_window_shape = pooling_window_shape
        self.number_of_strides = number_of_strides
        self.pooling_type = pooling_type.value
        self.padding_type = padding_type.value

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        input_rank = tf.rank(inputs)
        if input_rank == 3:
            inputs = inputs[:, :, :, tf.newaxis]
            pooled = tf.nn.pool(
                input=inputs,
                window_shape=self.pooling_window_shape,
                pooling_type=self.pooling_type,
                strides=self.number_of_strides,
                padding=self.padding_type
            )
            return pooled[0, :, :, 0]
        else:
            raise ValueError(f"Inputs are required to have rank 3, given {input_rank}")
