from pathlib import Path
from typing import Tuple

import pandas as pd
from metric_learn import NCA, LMNN

from src.candidate_reranker.CandidateReranker import CandidateReranker
from src.candidate_reranker.configurables.reranker_configurable_enums import LambdaEmbedderAttached, \
    TableEmbedderPoolingType, TableEmbeddingContent
from src.candidate_reranker.lambda_calculus_embedding import LambdaEmbedder
from src.candidate_reranker.table_embedding.TableEmbedder import TableEmbedder
from src.candidate_resolver import training as candidate_resolver_training
from src.candidate_resolver.CandidateResolver import CandidateResolver
from src.candidate_resolver.configurables.resolver_configurable_enums import BertKerasLayerType, MetricLearnerType, \
    EmbeddingType
from src.candidate_resolver.embedding.BertEmbedder import BertEmbedder
from src.candidate_resolver.embedding.embedder_functions import EmbeddingFunctionProvider
from src.entity_abstractor.Abstractor import Abstractor
from src.entity_abstractor.MockAbstractor import MockAbstractor
from src.entity_abstractor.configurables.abstractor_configurable_enums import AbstractionType


def get_resolver_representation_from_arguments(
        resolver_argument_pack: Tuple[BertKerasLayerType, EmbeddingType, MetricLearnerType]) -> str:
    return "({0}_{1}_metric_learner_{2})".format(
        resolver_argument_pack[0].name,
        resolver_argument_pack[1].name,
        resolver_argument_pack[2].name
    )


def get_reranker_representation_from_arguments(
        reranker_attached: bool,
        reranker_argument_pack: Tuple[
            Tuple[BertKerasLayerType, TableEmbedderPoolingType, TableEmbeddingContent],
            LambdaEmbedderAttached
        ]) -> str:
    table_input_pack, lambda_embedder_attached = reranker_argument_pack
    return "_reranker({0}_{1}_{2}_lambda_attached_{3})".format(
        table_input_pack[0].name,
        table_input_pack[1].name,
        table_input_pack[2].name,
        lambda_embedder_attached.name
    ) if reranker_attached else ""


def create_abstractor(abstraction_type: AbstractionType):
    return Abstractor() if abstraction_type == AbstractionType.DEPENDENCY else MockAbstractor()


def create_resolver_model_from(resolver_dataset: pd.DataFrame,
                               lifted_instance_column_name: str,
                               lifted_output_column_name: str,
                               embedding_function_provider: EmbeddingFunctionProvider,
                               utterance_embedding_type: EmbeddingType,
                               metric_learner_type: MetricLearnerType,
                               save_location: Path,
                               take_best_guess: bool = False):
    metric_learner = create_metric_learner(metric_learner_type)
    return candidate_resolver_training.create_and_train_model(
        resolver_dataset,
        lifted_instance_column_name, lifted_output_column_name,
        save_location,
        embedding_function_provider.select_embedding_function(utterance_embedding_type),
        metric_learner=metric_learner,
        regressor_learning_rate=1e-3,
        take_best_guess=take_best_guess
    )


def create_reranker_model_from(candidate_resolver: CandidateResolver,
                               reranker_dataset: pd.DataFrame,
                               table_embedder: TableEmbedder,
                               lambda_embedder_attached: LambdaEmbedderAttached):
    lambda_embedder = create_lambda_embedder(reranker_dataset, lambda_embedder_attached)
    return CandidateReranker.create_and_train(candidate_resolver,
                                              reranker_dataset,
                                              table_embedder,
                                              lambda_embedder=lambda_embedder)


def create_embedding_function_provider(bert_layer_type: BertKerasLayerType):
    embedder = BertEmbedder.initialize(
        "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
        bert_layer_type.value
    )
    return EmbeddingFunctionProvider(embedder)


def create_metric_learner(metric_learner_type: MetricLearnerType):
    if metric_learner_type == MetricLearnerType.NCA:
        metric_learner = NCA(max_iter=1000)
    elif metric_learner_type == MetricLearnerType.LMNN:
        metric_learner = LMNN(k=1, learn_rate=1e-3)
    else:
        metric_learner = None
    return metric_learner


def create_lambda_embedder(reranker_dataset: pd.DataFrame, lambda_embdedder_attached: LambdaEmbedderAttached):
    return LambdaEmbedder.Builder().set_lambda_embedder_data(reranker_dataset, "DSL output").build() \
        if lambda_embdedder_attached.value else None
