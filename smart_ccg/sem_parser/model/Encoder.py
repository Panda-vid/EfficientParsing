import tensorflow as tf
from tensorflow.keras import layers, losses


class Encoder(tf.keras.Sequential):
    def __init__(self, input_size, hidden_size, output_size):
        super(Encoder, self).__init__()
        # input_size is equal to the maximum sequence length which is usually 250
        # thus input_shape is usually (250,)
        self.add(layers.Dense(hidden_size,
                              input_shape=(input_size,),
                              activation='relu',
                              kernel_constraint=tf.keras.constraints.max_norm(3)))
        self.add(layers.Dropout(0.2))
        self.add(layers.Dense(output_size,
                              activation='relu',
                              kernel_constraint=tf.keras.constraints.max_norm(3)))
        self.compile(optimizer='adam', loss=losses.cosine_similarity)

    def call(self, inputs, training=None, mask=None):
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs
