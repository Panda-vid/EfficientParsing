import tensorflow as tf


class Regressor(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.regressor_layer = tf.keras.layers.Dense(
            1, activation="sigmoid", trainable=True, use_bias=True, bias_initializer=tf.keras.initializers.Constant(5.)
        )

    def call(self, inputs):
        return self.regressor_layer(inputs)
