import pickle
import bert

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
# needed for BERT/ELECTRA to work
import tensorflow_text as text
import pandas as pd

from pathlib import Path
from tensorflow.keras import layers, models

from smart_ccg.sem_parser.model.ScalarLogisticRegressor import ScalarLogisticRegressor
from smart_ccg.sem_parser.model.model_utils import generate_bert_preprocessor, positional_encoding, cosine_similarity, \
    get_last_nonzero_index


class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.bert_model_url = None
        self.bert_preprocess = None
        self.bert = None
        self.max_sequence_length = None
        self.encoder = None
        self.classifier = None
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
        model.bert_preprocess = generate_bert_preprocessor()
        model.max_sequence_length = max_sequence_length
        model.encoder = encoder
        model.classifier = classifier
        model.threshold = not_sure_threshold
        model.trained_examples = None
        model.positional_encoding = positional_encoding(max_sequence_length, bert_model_output_size)[0]
        return model

    def __call__(self, inputs):
        inputs = self.preprocess(inputs)
        inputs = self.encoder(inputs)
        distances = self.compute_distances(inputs)
        return self.predict_label(distances)

    def compute_distances(self, inputs):
        distances = np.array([])
        for index, row in self.trained_examples.iterrows():
            feature_vector = row["Lifted instance"]
            distances = np.append(distances, self.classifier(cosine_similarity(feature_vector, inputs)))
        return distances

    def predict_label(self, distances):
        closest = tf.math.top_k(distances, k=1)
        all_trained_labels = self.trained_examples["DSL output"]
        if closest.values.numpy()[0] >= self.threshold:
            index = closest.indices.numpy()[0]
            predicted_label = all_trained_labels[index]
        else:
            # TODO: introduce NOT_SURE label
            predicted_label = -1
        return predicted_label

    def preprocess(self, lifted_input: str):
        # ELECTRA/BERT preprocessing probably needs to be pulled out further for padding
        lifted_input = tf.constant([lifted_input], dtype=tf.string)
        bert_input = self.bert_preprocess(lifted_input)
        seq_length = get_last_nonzero_index(bert_input['input_mask'].numpy())
        # bert_output are the embedded vectors given by ELECTRA/BERT either as a sequence or as a fixed-sized vector
        bert_output = self.bert(bert_input)
        lifted_input = bert_output['sequence_output']
        lifted_input = tf.squeeze(lifted_input)[:seq_length, :]

        # further positional encoding of the sequence given
        lifted_input += self.positional_encoding[:seq_length]

        # padding the sequence such that the MLP can use the vector
        if seq_length <= self.max_sequence_length:
            lifted_input = tf.pad(lifted_input,
                                  tf.constant([[0,  int(self.max_sequence_length - seq_length)], [0, 0]], dtype=tf.int32),
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
        model.encoder = encoder
        model.classifier = classifier
        model.bert = hub.KerasLayer(model.bert_model_url, trainable=False)
        model.bert_preprocess = generate_bert_preprocessor()

        return model

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['encoder']
        del state['classifier']
        del state['bert_preprocess']
        del state['bert']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # these attributes need to be set while loading
        self.encoder = None
        self.classifier = None
        self.bert_preprocess = None
        self.tokenizer = None
        self.bert = None
        self.similarity = None


if __name__ == '__main__':
    model = Model.load(Path("../../../resources/models/small"))
    lifted_instance = "Select [,column]"
    prediction = model(lifted_instance)
    all_outputs = model.trained_examples["DSL output"]

    if prediction != -1:
        print(prediction)
    else:
        print("Not sure")
