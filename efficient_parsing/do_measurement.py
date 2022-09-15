"""
..
This script measures the accuracy of the semantic parser in different configurations.
We expose 8 measurements:
    - 'resolver_utterance_embedding' measures the impact on parsing performance of different text embedding models.
    - 'resolver_embedding_postprocessing' measures the impact of different postprocessing methods on parsing performance.
    - 'resolver_metric_learn' measures the impact feature clustering using metric learning on parsing performance.
    - 'reranker_table_embedding' measures the impact of including the reranker in the semantic parser pipeline as well as the performance impact of different table embedding language models
    - 'reranker_table_embedding_pooling' measures the impact of pooling reranker table embeddings on parsing performance.
    - 'reranker_lambda_embedding' measures the performance impact of including embedded lambda calculus parses of the original sentence in the reranker embeddings.
    - 'abstractor' compares the performance of our rule-based entity abstraction method vs. a perfect abstraction mechanism.

'all' does all measurements at once. We do not recommend running the script this way as this will take considerable time to compute.
"""

import argparse
import logging

from src.evaluation.measure_performance import resolver_utterance_measurement, \
    resolver_postprocessing_measurement, metric_learn_measurement, reranker_table_embedding_measurement, \
    reranker_table_embedding_pooling_measurement, reranker_lambda_embedder_attached_measurement, \
    abstractor_measurement, one_shot_generalization_test

parser = argparse.ArgumentParser(
    description="Measure the accuracy of the parser along different configuration axes."
)
parser.add_argument(
    "measurement", choices=[
    "all", "resolver_utterance_embedding", "resolver_embedding_postprocessing", "resolver_metric_learn",
    "reranker_table_embedding", "reranker_table_embedding_pooling", "reranker_lambda_embedding", "abstractor",
    "generalization"
    ], help="The type of measurement to be done on the system. Refer to the documentation for further information."
)


def resolve_arguments(input_arguments) -> None:
    """
    Chooses the correct measurements to be executed from the input arguments parsed by argparse.
    :param input_arguments: The argparse input arguments containing one field called 'measurement'
    :return: None
    """
    if input_arguments.measurement == "all":
        do_all_measurements()
    elif input_arguments.measurement == "resolver_utterance_embedding":
        do_resolver_utterance_measurement()
    elif input_arguments.measurement == "resolver_embedding_postprocessing":
        do_resolver_postprocessing_measurement()
    elif input_arguments.measurement == "resolver_metric_learn":
        do_resolver_postprocessing_measurement()
    elif input_arguments.measurement == "reranker_table_embedding":
        do_reranker_table_embedding_measurement()
    elif input_arguments.measurement == "reranker_table_embedding_pooling":
        do_reranker_table_embedding_pooling_measurement()
    elif input_arguments.measurement == "reranker_lambda_embedding":
        do_reranker_lambda_embedding_measurement()
    elif input_arguments.measurement == "abstractor":
        do_abstractor_measurement()
    elif input_arguments.measurement == "generalization":
        do_one_shot_generalization_measurement()


def do_all_measurements() -> None:
    """
    Run all available measurements.
    :return: None
    """
    logging.info("running all measurements")
    do_resolver_utterance_measurement()
    do_resolver_postprocessing_measurement()
    do_resolver_metric_learn_measurement()
    do_reranker_table_embedding_measurement()
    do_reranker_table_embedding_pooling_measurement()
    do_reranker_lambda_embedding_measurement()
    do_abstractor_measurement()
    do_one_shot_generalization_measurement()


def do_resolver_utterance_measurement() -> None:
    """
    Starts the resolver utterance embedding measurement from the thesis.
    :return: None
    """
    logging.info("running resolver utterance embedding measurement")
    resolver_utterance_measurement()


def do_resolver_postprocessing_measurement() -> None:
    """
    Starts the resolver utterance embedding postprocessing measurement from the thesis.
    :return: None
    """
    logging.info("running resolver utterance embedding postprocessing measurement")
    resolver_postprocessing_measurement()


def do_resolver_metric_learn_measurement() -> None:
    """
    Starts the resolver metric learning measurement from the thesis.
    :return: None
    """
    logging.info("running resolver embedding postprocessing measurement")
    metric_learn_measurement()


def do_reranker_table_embedding_measurement() -> None:
    """
    Starts the reranker table embedding measurement from the thesis.
    :return: None
    """
    logging.info("running reranker table embedding measurement")
    reranker_table_embedding_measurement()


def do_reranker_table_embedding_pooling_measurement() -> None:
    """
    Starts the reranker table embedding pooling measurement from the thesis.
    :return: None
    """
    logging.info("running reranker table embedding pooling measurement")
    reranker_table_embedding_pooling_measurement()


def do_reranker_lambda_embedding_measurement() -> None:
    """
    Starts the reranker lambda embedding measurement from the thesis.
    :return: None
    """
    logging.info("running reranker lambda embedding measurement")
    reranker_lambda_embedder_attached_measurement()


def do_abstractor_measurement() -> None:
    """
    Starts the abstractor measurement from the thesis.
    :return: None
    """
    logging.info("running abstractor measurement")
    abstractor_measurement()


def do_one_shot_generalization_measurement() -> None:
    """
    Starts the one-shot generalization measurement from the thesis.
    :return: None
    """
    logging.info("running one-shot generalization measurement")
    one_shot_generalization_test()


if __name__ == '__main__':
    args = parser.parse_args()
    resolve_arguments(args)

