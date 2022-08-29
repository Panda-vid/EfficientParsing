import tensorflow as tf

from typing import List


def join(tensors: List[tf.Tensor], separator: tf.Tensor) -> tf.Tensor:
    res = None
    for i, tensor in enumerate(tensors):
        pack = tf.concat([tensor, separator], 1) if i + 1 < len(tensors) else tensor
        res = pack if i == 0 else tf.concat([res, pack], 1)
    return res
