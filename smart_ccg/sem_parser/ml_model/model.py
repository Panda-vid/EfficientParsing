import itertools
import pickle
from pathlib import Path
from typing import Generator

import bert
import numpy as np
import tensorflow as tf
import pandas as pd
# needed for BERT/ELECTRA to work
import tensorflow_text
from tensorflow.keras import layers, models
import tensorflow_hub as hub
from tqdm import trange


class Model:
    def __init__(self):
        self.bert_model_url = None
        self.preprocess = None
        self.tokenizer = None
        self.bert = None
        self.max_sequence_length = None
        self.encoder = None
        self.classifier = None
        self.similarity = tf.keras.metrics.CosineSimilarity()
        self.threshold = None
        self.trained_examples = None
        self.positional_encoding = None

    @classmethod
    def create_model_from(cls, encoder, classifier, not_sure_threshold, bert_model_url,
                          bert_model_output_size, max_sequence_length):
        model = cls()
        # Load the BERT/ELECTRA encoder and preprocessing models
        # example bert: hub.load('https://tfhub.dev/google/electra_base/2')
        model.bert_model_url = bert_model_url
        model.bert = hub.KerasLayer(bert_model_url, trainable=False)
        # example preprocess: hub.load('https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3')
        model.preprocess = hub.load('https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3')
        # tokenizer initialization
        BertTokenizer = bert.bert_tokenization.FullTokenizer
        vocabulary_file = model.bert.resolved_object.vocab_file.asset_path.numpy()
        to_lower_case = model.bert.resolved_object.do_lower_case.numpy()
        model.tokenizer = BertTokenizer(vocabulary_file, to_lower_case)

        model.max_sequence_length = max_sequence_length
        model.encoder = encoder
        model.classifier = classifier
        model.threshold = not_sure_threshold
        model.trained_examples = None
        model.positional_encoding = positional_encoding(max_sequence_length, bert_model_output_size)[0]
        return model

    def __call__(self, inputs):
        inputs = self.pre_encoding(inputs)
        inputs = self.encoder(inputs)
        distances = np.array([])
        for index, row in self.trained_examples.iterrows():
            feature_vector = row["Lifted instance"]
            self.similarity.update_state(inputs, feature_vector)
            np.append(distances, self.classifier(self.similarity.result().numpy()))
            self.similarity.reset_state()
        closest = tf.math.top_k(distances, k=1)
        all_trained_labels = self.trained_examples["DSL output"]
        if closest.values.numpy()[0] <= self.threshold:
            index = closest.indices.numpy()[0]
            predicted_label = all_trained_labels[index]
        else:
            # TODO: introduce NOT_SURE label
            predicted_label = 0

        return predicted_label

    def pre_encoding(self, lifted_input):
        # ELECTRA/BERT preprocessing probably needs to be pulled out further for padding
        tokenized_input = self.tokenizer.tokenize(lifted_input)
        bert_input = self.preprocess(tokenized_input)
        seq_length = tf.shape(bert_input['input_mask'])[0].numpy()
        # bert_output are the embedded vectors given by ELECTRA/BERT either as a sequence or as a fixed-sized vector
        bert_output = self.bert(bert_input)
        lifted_input = bert_output['pooled_output']
        # further positional encoding of the sequence given
        lifted_input += self.positional_encoding[:seq_length]

        # padding the sequence such that the MLP can use the vector
        if seq_length <= self.max_sequence_length:
            lifted_input = tf.pad(lifted_input, tf.constant([[0,  self.max_sequence_length - seq_length], [0, 0]]),
                                  "CONSTANT")
        else:
            raise RuntimeError("The given sequence is too long for the model to handle!")

        # non-linear transform of encoder output (kernel method with quadratic kernel)
        lifted_input = tf.tensordot(lifted_input, tf.transpose(lifted_input, [1, 0]), axes=1) ** 2

        pre_encoded_input = tf.math.reduce_sum(lifted_input, axis=1)
        return pre_encoded_input[None, :]

    def set_training_examples(self, dataframe: pd.DataFrame):
        self.trained_examples = dataframe

    def save(self, model_folder_path: Path):
        model_path = model_folder_path / "model.pkl"
        encoder_path = model_folder_path / "encoder.model"
        classifier_path = model_folder_path / "classifier.model"
        pickle.dump(self, model_path.open("wb"), pickle.HIGHEST_PROTOCOL)
        self.encoder.save(str(encoder_path))
        self.classifier.save(classifier_path)
        del self

    @classmethod
    def load(cls, model_folder_path: Path):
        if model_folder_path.is_dir():
            model_path = model_folder_path / "model.pkl"
            encoder_path = model_folder_path / "encoder.model"
            classifier_path = model_folder_path / "classifier.model"
        else:
            raise RuntimeError("Directory {} not found".format(str(model_folder_path)))

        encoder = models.load_model(str(encoder_path))
        classifier = ScalarLogisticRegressor.load(classifier_path)
        model = pickle.load(model_path.open("rb"))
        model.similarity = tf.keras.metrics.CosineSimilarity()
        model.encoder = encoder
        model.classifier = classifier
        model.bert = hub.KerasLayer(model.bert_model_url)
        model.preprocess = hub.load('https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3')
        BertTokenizer = bert.bert_tokenization.FullTokenizer
        vocabulary_file = model.bert.resolved_object.vocab_file.asset_path.numpy()
        to_lower_case = model.bert.resolved_object.do_lower_case.numpy()
        model.tokenizer = BertTokenizer(vocabulary_file, to_lower_case)

        return model

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['encoder']
        del state['classifier']
        del state['preprocess']
        del state['tokenizer']
        del state['bert']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # these attributes need to be set while loading
        self.encoder = None
        self.classifier = None
        self.preprocess = None
        self.tokenizer = None
        self.bert = None
        self.similarity = None


class ScalarLogisticRegressor:
    def __init__(self, seed=18):
        initializer = tf.random_uniform_initializer(seed=seed)
        self.a = tf.Variable(tf.squeeze(initializer(shape=[1], dtype=tf.float32)))
        self.b = tf.Variable(tf.squeeze(initializer(shape=[1], dtype=tf.float32)))
        self.activation = tf.math.sigmoid
        self.optimizer = tf.keras.optimizers.Adam()
        self.loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    def train_from_generator(self, generator: Generator, epochs=30):
        x_batch_iter, y_batch_iter = itertools.tee(iter(generator))
        x_batch = tf.convert_to_tensor([x[0] for x in x_batch_iter])
        y_batch = tf.convert_to_tensor([y[1] for y in y_batch_iter])
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
        x_dim = x_batch[0].shape.rank
        if not x_dim == 0:
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


class Encoder(tf.keras.Sequential):
    def __init__(self, input_size, hidden_size, output_size):
        super(Encoder, self).__init__()
        # input_size is equal to the maximum sequence length which is usually 250
        # thus input_shape is usually (250,)
        self.add(layers.Dense(hidden_size, input_shape=(input_size,), activation='relu'))
        self.add(layers.Dense(output_size, activation='relu'))
        self.compile(optimizer='adam', loss=tf.keras.losses.cosine_similarity)

    def call(self, inputs, training=None, mask=None):
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs


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
