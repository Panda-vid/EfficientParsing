"""
This module contains the setups for each performance measurement from the thesis and is the interface for :py:module:`~efficient_parsing.do_measurement`.
"""

from src.candidate_reranker.configurables.reranker_configurable_enums import TableEmbedderPoolingType, \
    TableEmbeddingContent, LambdaEmbedderAttached
from src.candidate_resolver.configurables.resolver_configurable_enums import EmbeddingType, MetricLearnerType, \
    BertKerasLayerType
from src.entity_abstractor.Abstractor import Abstractor
from src.entity_abstractor.configurables.abstractor_configurable_enums import AbstractionType
from src.evaluation.Measurement import Measurement


def resolver_utterance_measurement():
    """
    Provides the setup for the resolver utterance measurement and runs it.
    :return: None
    """
    data_difficulties = [1, 2, 3]
    reranker_attached = [False]
    resolver_embedding_types = [EmbeddingType.MAX_POOLED]
    metric_learner_types = [MetricLearnerType.NONE]
    bert_layer_types = [bert_layer_type for bert_layer_type in BertKerasLayerType]
    measurement = Measurement(
        "resolver_utterance_embedding_measurement", dataset_difficulties=data_difficulties,
        reranker_attached_configurations=reranker_attached, resolver_embedding_types=resolver_embedding_types,
        bert_layer_types=bert_layer_types, metric_learner_types=metric_learner_types
    )
    measurement.run_tests()


def resolver_postprocessing_measurement():
    """
    Provides the setup for the resolver utterance postprocessing measurement and runs it.
    :return: None
    """
    data_difficulties = [1, 2, 3]
    reranker_attached = [False]
    resolver_embedding_types = [embedding_type for embedding_type in EmbeddingType]
    metric_learner_types = [MetricLearnerType.NONE]
    bert_layer_types = [BertKerasLayerType.BERT_SMALL]
    measurement = Measurement(
        "resolver_embedding_postprocessing_measurement", dataset_difficulties=data_difficulties,
        reranker_attached_configurations=reranker_attached, resolver_embedding_types=resolver_embedding_types,
        metric_learner_types=metric_learner_types, bert_layer_types=bert_layer_types
    )
    measurement.run_tests()


def metric_learn_measurement():
    """
    Provides the setup for the resolver metric learning measurement and runs it.
    :return: None
    """
    data_difficulties = [1, 2, 3]
    reranker_attached = [False]
    resolver_embedding_types = [EmbeddingType.MAX_POOLED]
    metric_learner_types = [metric_learner_type for metric_learner_type in MetricLearnerType]
    bert_layer_types = [BertKerasLayerType.BERT_SMALL]
    measurement = Measurement(
        "metric_learn_measurement", dataset_difficulties=data_difficulties,
        reranker_attached_configurations=reranker_attached, resolver_embedding_types=resolver_embedding_types,
        metric_learner_types=metric_learner_types, bert_layer_types=bert_layer_types
    )
    measurement.run_tests()


def reranker_table_embedding_measurement():
    """
    Provides the setup for the reranker table embedding measurement and runs it.
    :return: None
    """
    data_difficulties = [1, 2, 3]
    reranker_attached = [True]
    lambda_embedder_attached = [LambdaEmbedderAttached.NO]
    metric_learner_types = [MetricLearnerType.NONE]
    resolver_embedding_types = [EmbeddingType.MAX_POOLED]
    bert_layer_types = [BertKerasLayerType.BERT_SMALL]
    table_embedding_bert_layer_types = [bert_layer_type for bert_layer_type in BertKerasLayerType]
    table_embedding_pooling_types = [TableEmbedderPoolingType.MAX]
    table_embedding_contents = [TableEmbeddingContent.COLUMN_ONLY]
    measurement = Measurement(
        "reranker_table_embedding_measurement", dataset_difficulties=data_difficulties,
        reranker_attached_configurations=reranker_attached, resolver_embedding_types=resolver_embedding_types,
        metric_learner_types=metric_learner_types, bert_layer_types=bert_layer_types,
        table_embedder_bert_layer_types=table_embedding_bert_layer_types,
        table_embedder_pooling_types=table_embedding_pooling_types,
        table_embedding_content_configurations=table_embedding_contents,
        reranker_lambda_embedder_attachments=lambda_embedder_attached
    )

    measurement.run_tests()


def reranker_table_embedding_pooling_measurement():
    """
    Provides the setup for the reranker table embedding pooling measurement and runs it.
    :return: None
    """
    data_difficulties = [1, 2, 3]
    reranker_attached = [True]
    lambda_embedder_attached = [LambdaEmbedderAttached.NO]
    metric_learner_types = [MetricLearnerType.NONE]
    resolver_embedding_types = [EmbeddingType.MAX_POOLED]
    bert_layer_types = [BertKerasLayerType.BERT_SMALL]
    table_embedding_bert_layer_types = [BertKerasLayerType.BERT_SMALL]
    table_embedding_pooling_types = [
        table_embedding_pooling_type for table_embedding_pooling_type in TableEmbedderPoolingType
    ]
    table_embedding_contents = [TableEmbeddingContent.COLUMN_ONLY]

    measurement = Measurement(
        "reranker_table_embedding_pooling_measurement", dataset_difficulties=data_difficulties,
        reranker_attached_configurations=reranker_attached, resolver_embedding_types=resolver_embedding_types,
        metric_learner_types=metric_learner_types, bert_layer_types=bert_layer_types,
        table_embedder_bert_layer_types=table_embedding_bert_layer_types,
        table_embedder_pooling_types=table_embedding_pooling_types,
        table_embedding_content_configurations=table_embedding_contents,
        reranker_lambda_embedder_attachments=lambda_embedder_attached
    )

    measurement.run_tests()


def reranker_lambda_embedder_attached_measurement():
    """
    Provides the setup for the reranker lambda embedding measurement and runs it.
    :return: None
    """
    data_difficulties = [1, 2, 3]
    reranker_attached = [True]
    lambda_embedder_attached = [LambdaEmbedderAttached.YES]
    metric_learner_types = [MetricLearnerType.NONE]
    resolver_embedding_types = [EmbeddingType.MAX_POOLED]
    bert_layer_types = [BertKerasLayerType.BERT_SMALL]
    table_embedding_bert_layer_types = [BertKerasLayerType.BERT_SMALL]
    table_embedding_pooling_types = [TableEmbedderPoolingType.MAX]
    table_embedding_contents = [TableEmbeddingContent.COLUMN_ONLY]

    measurement = Measurement(
        "reranker_lambda_embedder_attached_measurement", dataset_difficulties=data_difficulties,
        reranker_attached_configurations=reranker_attached, resolver_embedding_types=resolver_embedding_types,
        metric_learner_types=metric_learner_types, bert_layer_types=bert_layer_types,
        table_embedder_bert_layer_types=table_embedding_bert_layer_types,
        table_embedder_pooling_types=table_embedding_pooling_types,
        table_embedding_content_configurations=table_embedding_contents,
        reranker_lambda_embedder_attachments=lambda_embedder_attached,
        test_repetitions=5
    )

    measurement.run_tests()


def abstractor_measurement():
    """
    Provides the setup for the abstractor measurement and runs it.
    :return: None
    """
    Abstractor.start_core_nlp_server()
    Abstractor.wait_until_server_reachable()
    data_difficulties = [1, 2, 3]
    reranker_attached = [False]
    bert_layer_types = [BertKerasLayerType.ELECTRA_LARGE]
    resolver_embedding_types = [EmbeddingType.AVG_POOLED_POSITIONAL]
    metric_learner_types = [MetricLearnerType.LMNN]
    abstractor_types = [AbstractionType.DEPENDENCY, AbstractionType.MOCK]
    measurement = Measurement(
        "abstractor_measurement", dataset_difficulties=data_difficulties,
        reranker_attached_configurations=reranker_attached, resolver_embedding_types=resolver_embedding_types,
        metric_learner_types=metric_learner_types, bert_layer_types=bert_layer_types,
        abstractor_configurations=abstractor_types, resolver_relative_not_sure_threshold=4
    )

    measurement.run_tests()
    Abstractor.stop_core_nlp_server()


def one_shot_generalization_test():
    """
    Provides the setup for the one-shot generalization measurement and runs it.
    :return: None
    """
    data_difficulties = [3]
    reranker_attached = [False]
    bert_layer_types = [BertKerasLayerType.BERT_LARGE, BertKerasLayerType.ELECTRA_LARGE, BertKerasLayerType.BERT_SMALL]
    resolver_embedding_types = [EmbeddingType.AVG_POOLED_POSITIONAL]
    metric_learner_types = [MetricLearnerType.LMNN, MetricLearnerType.NONE]
    abstractor_types = [AbstractionType.MOCK]
    test_repetitions = 1

    measurement = Measurement(
        "one_shot_generalization_measurement", dataset_difficulties=data_difficulties,
        reranker_attached_configurations=reranker_attached, resolver_embedding_types=resolver_embedding_types,
        metric_learner_types=metric_learner_types, bert_layer_types=bert_layer_types,
        abstractor_configurations=abstractor_types, test_repetitions=test_repetitions,
        one_shot_generalization_test=True, resolver_relative_not_sure_threshold=4
    )

    measurement.run_tests()
