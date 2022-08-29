import tensorflow as tf

from typing import Tuple


class RerankerNetwork(tf.keras.Model):
    def __init__(self,
                 input_shape: Tuple,
                 hidden_layer_dim: int,
                 output_layer_dim: int,
                 dropout_hidden_layer: float,
                 dropout_output_layer: float,
                 num_epochs: int = 2,
                 learning_rate: float = 1e-3):
        super().__init__()
        self.reranker_input_shape = input_shape
        self.hidden_layer_dim = hidden_layer_dim
        self.output_layer_dim = output_layer_dim
        self.dropout_hidden_layer = dropout_hidden_layer
        self.dropout_output_layer = dropout_output_layer
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.dropout_hidden = tf.keras.layers.Dropout(dropout_hidden_layer, input_shape=input_shape)
        self.hidden_layer = tf.keras.layers.Dense(hidden_layer_dim, activation="relu")
        self.dropout_output = tf.keras.layers.Dropout(dropout_output_layer)
        self.output_layer = tf.keras.layers.Dense(output_layer_dim, activation="sigmoid")

    @classmethod
    def create_and_train(cls,
                         reranker_features,
                         reranker_labels,
                         input_shape: Tuple,
                         hidden_layer_dim: int,
                         output_layer_dim: int,
                         dropout_hidden_layer: float,
                         dropout_output_layer: float,
                         num_epochs: int = 2,
                         learning_rate: float = 1e-3):
        reranker = cls(
            input_shape, hidden_layer_dim,
            output_layer_dim, dropout_hidden_layer,
            dropout_output_layer, num_epochs, learning_rate
        )
        reranker.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                         loss=tf.keras.losses.BinaryCrossentropy(),
                         metrics=[tf.keras.metrics.BinaryCrossentropy()])
        reranker.fit(reranker_features, reranker_labels, epochs=num_epochs)
        return reranker

    def call(self, inputs, training=False):
        inputs = self.dropout_hidden(inputs, training=training)
        inputs = self.hidden_layer(inputs)
        inputs = self.dropout_output(inputs, training=training)
        return self.output_layer(inputs)

    def get_hyperparams(self) -> Tuple[float, float, int, float]:
        return self.dropout_hidden, self.dropout_output_layer, self.num_epochs, self.learning_rate
