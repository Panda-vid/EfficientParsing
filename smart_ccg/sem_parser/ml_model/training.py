import itertools
import argparse
from enum import Enum
from pathlib import Path

import pandas as pd
import tensorflow as tf
from smart_ccg.sem_parser.ml_model.model import Encoder, Model, ScalarLogisticRegressor


class ElectraType(Enum):
    SMALL = ("https://tfhub.dev/google/electra_small/2", 256)
    MEDIUM = ("https://tfhub.dev/google/electra_base/2", 768)
    LARGE = ("https://tfhub.dev/google/electra_large/2", 1024)


def read_initial_dataset_from(csv_file):
    return pd.read_csv(csv_file, sep=";")


def initialize_model(not_sure_threshold, bert_model_type, encoder_hidden_size, encoder_output_size,
                     max_sequence_length=250):
    classifier = ScalarLogisticRegressor()
    bert_model_output_size = bert_model_type.value[1]
    encoder = Encoder(max_sequence_length, encoder_hidden_size, encoder_output_size)
    return Model.create_model_from(encoder, classifier, not_sure_threshold, bert_model_type.value[0],
                                   bert_model_output_size, max_sequence_length)


def train_model(model: Model, dataframe, encoder_num_epochs, classifier_num_epochs, model_directory_path: Path):
    dataframe = pre_encode_dataset(dataframe, model)
    model = train_encoder(dataframe, model, encoder_num_epochs)
    model = train_classifier(dataframe, model, classifier_num_epochs)
    dataframe["Lifted instance"] = dataframe["Lifted instance"].apply(model.encoder)
    model.set_training_examples(dataframe)

    if not model_directory_path.is_dir():
        model_directory_path.mkdir()
    model.save(model_directory_path)


def pre_encode_dataset(dataframe, model: Model):
    dataframe["Lifted instance"] = dataframe["Lifted instance"].apply(model.pre_encoding)
    return dataframe


def train_classifier(dataframe, model: Model, num_epochs):
    model.classifier.train_from_generator(classifier_pair_generator(dataframe, model), num_epochs)
    return model


def classifier_pair_generator(dataframe, model: Model):
    dataframe_index_pairs = itertools.combinations(dataframe.index, 2)
    for id1, id2 in dataframe_index_pairs:
        encoded_instance_id1 = model.encoder(dataframe.loc[id1, "Lifted instance"])
        encoded_instance_id2 = model.encoder(dataframe.loc[id2, "Lifted instance"])

        model.similarity.update_state(encoded_instance_id1, encoded_instance_id2)
        cosine_similarity = model.similarity.result()
        model.similarity.reset_state()

        has_same_DSL_output = dataframe.loc[id1, "DSL output"] == dataframe.loc[id2, "DSL output"]

        yield cosine_similarity, float(1) if has_same_DSL_output else cosine_similarity, float(0)


def train_encoder(dataframe, model: Model, num_epochs):
    model.encoder.fit(encoder_pair_generator(dataframe, model.encoder, num_epochs),
                      steps_per_epoch=len(dataframe.index), epochs=num_epochs, verbose=1)
    return model


def encoder_pair_generator(dataframe, encoder, num_passes):
    grouped_dict = get_all_examples_grouped_by_label(dataframe)
    # num_passes is used for repeating the input set for training such that the fit function does not run out of data
    for i in range(num_passes):
        for label, examples in grouped_dict.items():
            for example_1, example_2 in itertools.combinations(examples, 2):
                encoded_example = encoder(example_2)
                yield example_1, encoded_example


def get_all_examples_grouped_by_label(dataframe):
    unique_labels = get_unique_labels(dataframe)
    result = {}
    for label in unique_labels:
        result[label] = dataframe[dataframe["DSL output"] == label]["Lifted instance"]
    return result


def get_unique_labels(dataframe):
    return dataframe["DSL output"].unique()


parser = argparse.ArgumentParser(description="This script trains the model and saves the model implemented in " +
                                             "'model.py'.")
parser.add_argument("model_directory", type=str, help="This is the directory, where the model files should be " +
                                                      "saved for later use.")
parser.add_argument("dataset_location", type=str, help="The location of the dataset for training.")
parser.add_argument("not_sure_threshold", type=float, help="This value sets the distance at which the model outputs " +
                                                           "that it is not sure.")
parser.add_argument("encoder_hidden_size", type=int, help="This defines the dimension of the first layer of the " +
                                                          "encoder.")
parser.add_argument("encoder_output_size", type=int, help="This defines the dimension of the output of the " +
                                                          "second layer of the encoder.")
parser.add_argument("encoder_num_epochs", type=int, help="Number of training epochs for the encoder.")
parser.add_argument("classifier_num_epochs", type=int, help="Number of training epochs for the classifier.")
parser.add_argument("max_seq_length", nargs="?", type=int, default=250,
                    help="This defines the longest sequence the model should accept. This value is needed internally " +
                         "for padding the pre-encodings.")
parser.add_argument("electra_model", nargs="?", choices=["small", "medium", "large"], default="medium",
                    help="This defines which ELECTRA model to use. The default is 'medium' which maps to ELECTRA base.")

if __name__ == '__main__':
    args = parser.parse_args()
    model_directory_path = Path(args.model_directory)

    electra_model_type = ElectraType.MEDIUM
    if args.electra_model == "small":
        electra_model_type = ElectraType.SMALL
    if args.electra_model == "large":
        electra_model_type = ElectraType.LARGE

    dataset = read_initial_dataset_from(args.dataset_location)
    model = initialize_model(args.not_sure_threshold, electra_model_type, args.encoder_hidden_size,
                             args.encoder_output_size, args.max_seq_length)
    train_model(model, dataset, args.encoder_num_epochs, args.classifier_num_epochs, model_directory_path)




