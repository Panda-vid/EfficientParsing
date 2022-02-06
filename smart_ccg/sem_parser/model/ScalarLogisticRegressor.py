import pickle

import tensorflow as tf
import numpy as np

from tqdm import trange
from pathlib import Path


class ScalarLogisticRegressor:
    def __init__(self, seed=18):
        initializer = tf.random_uniform_initializer(seed=seed)
        self.a = tf.Variable(tf.squeeze(initializer(shape=[1], dtype=tf.float32)))
        self.b = tf.Variable(tf.squeeze(initializer(shape=[1], dtype=tf.float32)))
        self.activation = tf.math.sigmoid
        self.optimizer = tf.keras.optimizers.Adam()
        self.loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    def train_from_array(self, value_pairs: np.ndarray, epochs=30):
        x_batch = value_pairs[:, 0]
        y_batch = value_pairs[:, 1]
        self.train(x_batch, y_batch, epochs)

    def train(self, x_batch, y_batch, num_epochs):
        self.check_inputs(x_batch, y_batch)
        epoch_bar = trange(num_epochs, desc="Epochs")
        for i in epoch_bar:
            losses = self.optimize(x_batch, y_batch)
            epoch_bar.set_postfix_str("mean loss: {}".format(np.mean(losses)), refresh=True)

    def check_inputs(self, x_batch, y_batch):
        x_batch_size = x_batch.shape[0]
        y_batch_size = y_batch.shape[0]
        if not x_batch_size == y_batch_size:
            ValueError("The batches for both the features and labels need to be the same size.\n" +
                       "Given shapes: feature {}, label {}".format(x_batch.shape, y_batch.shape))
        if not len(x_batch.shape) == 1:
            ValueError("The given feature batch does not consist of scalars, "
                       "but this regressor can only take scalar inputs.\nGiven shape: {}".format(x_batch.shape))

    def optimize(self, inputs, labels):
        is_valid_label = lambda label: label not in [0, 1]
        if all([is_valid_label(label) for label in labels]):
            raise ValueError("This regressor can only take either '0' or '1' as label.\n Given: {}".format(labels))
        with tf.GradientTape() as tape:
            predictions = self(inputs)
            losses = self.loss(predictions, labels)

        gradients = tape.gradient(losses, [self.a, self.b])
        self.optimizer.apply_gradients(zip(gradients, [self.a, self.b]))
        return losses.numpy()

    def __call__(self, inputs):
        return self.apply(inputs)

    def apply(self, inputs):
        if isinstance(inputs, np.ndarray):
            inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
        return self.activation(tf.math.add(tf.math.scalar_mul(self.a, inputs), self.b))

    def save(self, model_path: Path):
        pickle.dump(self, model_path.open("wb"), pickle.HIGHEST_PROTOCOL)
        del self

    @classmethod
    def load(cls, model_path: Path):
        model = pickle.load(model_path.open('rb'))
        model.activation = tf.math.sigmoid
        model.optimizer = tf.keras.optimizers.Adam()
        model.loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        return model

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['activation']
        del state['optimizer']
        del state['loss']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.activation = None
        self.optimizer = None
        self.loss = None
