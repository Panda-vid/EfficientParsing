import argparse
import logging

from src.evaluation.measure_performance import resolver_utterance_measurement, \
    resolver_postprocessing_measurement, metric_learn_measurement, reranker_table_embedding_measurement, \
    reranker_table_embedding_pooling_measurement, reranker_lambda_embedder_attached_measurement, abstractor_measurement,\
    one_shot_generalization_test

parser = argparse.ArgumentParser(
    description="Measure the accuracy of the parser in different along different configuration axes."
)
parser.add_argument(
    "measurement", choices=[
    "all", "resolver_utterance_embedding", "resolver_embedding_postprocessing", "resolver_metric_learn",
    "reranker_table_embedding", "reranker_table_embedding_pooling", "reranker_lambda_embedding", "abstractor"
    "generalization"
    ], help="The type of measurement to be done on the system. Refer to the documentation for further information."
)


def resolve_arguments(input_arguments):
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
        do_reranker_lambda_embedding_measurement()
    elif input_arguments.measurement == "generalization":
        do_one_shot_generalization_measurement()


def do_all_measurements():
    logging.info("running all measurements")
    do_resolver_utterance_measurement()
    do_resolver_postprocessing_measurement()
    do_resolver_metric_learn_measurement()
    do_reranker_table_embedding_measurement()
    do_reranker_table_embedding_pooling_measurement()
    do_reranker_lambda_embedding_measurement()
    do_abstractor_measurement()
    do_one_shot_generalization_measurement()


def do_resolver_utterance_measurement():
    logging.info("running resolver utterance embedding measurement")
    resolver_utterance_measurement()


def do_resolver_postprocessing_measurement():
    logging.info("running resolver utterance embedding postprocessing measurement")
    resolver_postprocessing_measurement()


def do_resolver_metric_learn_measurement():
    logging.info("running resolver embedding postprocessing measurement")
    metric_learn_measurement()


def do_reranker_table_embedding_measurement():
    logging.info("running reranker table embedding measurement")
    reranker_table_embedding_measurement()


def do_reranker_table_embedding_pooling_measurement():
    logging.info("running reranker table embedding pooling measurement")
    reranker_table_embedding_pooling_measurement()


def do_reranker_lambda_embedding_measurement():
    logging.info("running reranker lambda embedding measurement")
    reranker_lambda_embedder_attached_measurement()


def do_abstractor_measurement():
    logging.info("running abstractor measurement")
    abstractor_measurement()


def do_one_shot_generalization_measurement():
    logging.info("running one-shot generalization measurement")
    one_shot_generalization_test()


if __name__ == '__main__':
    args = parser.parse_args()
    resolve_arguments(args)

