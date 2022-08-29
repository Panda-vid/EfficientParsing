import itertools

from src.candidate_resolver.configurables.resolver_configurable_enums import BertKerasLayerType, MetricLearnerType, \
    EmbeddingType
from src.entity_abstractor.configurables.abstractor_configurable_enums import AbstractionType


def create_test_pipelines():
    argument_configurations = get_all_argument_configurations()


def get_all_argument_configurations():
    all_abstractor_configurations = [abstraction_type for abstraction_type in AbstractionType]
    all_bert_keras_layer_configurations = [
        (keras_layer_type_resolver, keras_layer_type_reranker)
        for keras_layer_type_resolver, keras_layer_type_reranker in itertools.permutations(BertKerasLayerType, 2)
    ]


def get_all_resolver_argument_configurations():
    all_metric_learner_configurations = [metric_learner_type for metric_learner_type in MetricLearnerType]
    all_embedding_types = [embedding_type for embedding_type in EmbeddingType]
