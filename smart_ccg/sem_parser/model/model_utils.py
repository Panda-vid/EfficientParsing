import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import numpy as np


def generate_bert_preprocessor() -> tf.keras.Model:
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
    preprocessor = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
    encoder_inputs = preprocessor(text_input)
    return tf.keras.Model(inputs=text_input, outputs=encoder_inputs, name="bert_preprocessor")


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(max_position, d_model):
    angle_rads = get_angles(np.arange(max_position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    # apply sin to even indices in the array
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # apply cos to odd indices in the array
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def cosine_similarity(x, y):
    x = np.array(x)
    y = np.array(y)
    return np.abs((x @ y.T)/(np.linalg.norm(x) * np.linalg.norm(y.T)).flatten())


def get_last_nonzero_index(tensor: tf.Tensor) -> int:
    zero = tf.constant(0, dtype=tf.int32)
    where = tf.not_equal(tensor, zero)
    indices = tf.where(where)
    return tf.shape(indices)[0]

