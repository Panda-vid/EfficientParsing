import numpy as np
import tensorflow as tf

from sklearn import metrics


def compute_cosine_similarities(class_vectors):
    return np.array(metrics.pairwise.cosine_similarity(class_vectors, class_vectors))


def cosine_similarity(x, y) -> float:
    x = np.array(x)
    y = np.array(y)
    return float((x @ y.T)/(np.linalg.norm(x) * np.linalg.norm(y.T)).flatten())


def summed_quadratic_kernel(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    return tf.reduce_sum(
        tf.tensordot(
            tf.squeeze(x) if len(x.shape) > 2 else tf.transpose(x),
            tf.transpose(tf.squeeze(y)) if len(y.shape) > 2 else y, axes=1
        )**2, 0
    )


def apply_positional_encoding(sequence_output: tf.Tensor, input_mask: tf.Tensor) -> tf.Tensor:
    sequence_length = get_last_nonzero_index(input_mask.numpy())
    max_sequence_length = sequence_output.shape[1]
    embedded_sequence = sequence_output[:, sequence_length, :]
    embedded_sequence += positional_encoding(
        max_sequence_length,
        sequence_output.shape[2]
    )[:, :sequence_length, :]
    return tf.pad(embedded_sequence,
                  tf.constant([
                      [0, 0],
                      [0, int(max_sequence_length - sequence_length)],
                      [0, 0]
                  ], dtype=tf.int32),
                  "CONSTANT")


def get_last_nonzero_index(tensor: tf.Tensor) -> int:
    zero = tf.constant(0, dtype=tf.int32)
    where = tf.not_equal(tensor, zero)
    indices = tf.where(where)
    return tf.shape(indices)[0]


def positional_encoding(max_position: int, d_model: int) -> tf.Tensor:
    angle_rates = get_angles(np.arange(max_position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
    # apply sin to even indices in the array
    angle_rates[:, 0::2] = np.sin(angle_rates[:, 0::2])
    # apply cos to odd indices in the array
    angle_rates[:, 1::2] = np.cos(angle_rates[:, 1::2])
    pos_encoding = angle_rates[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def get_angles(pos: int, i: int, d_model: int) -> np.ndarray:
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates
