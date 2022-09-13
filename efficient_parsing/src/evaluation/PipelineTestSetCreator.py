from pathlib import Path
from typing import List, Tuple, Iterable

import pandas as pd
from tqdm import tqdm

from src.candidate_reranker.CandidateReranker import CandidateReranker
from src.candidate_reranker.configurables.reranker_configurable_enums import TableEmbedderPoolingType, \
    TableEmbeddingContent, LambdaEmbedderAttached
from src.candidate_reranker.table_embedding.TableEmbedder import TableEmbedder
from src.candidate_resolver.CandidateResolver import CandidateResolver
from src.candidate_resolver.configurables.resolver_configurable_enums import EmbeddingType, MetricLearnerType, \
    BertKerasLayerType
from src.entity_abstractor.configurables.abstractor_configurable_enums import AbstractionType
from src.evaluation.test_set_creation_utils import create_abstractor, create_resolver_model_from, \
    get_resolver_representation_from_arguments, get_reranker_representation_from_arguments, create_reranker_model_from, \
    create_embedding_function_provider
from src.pipeline.SemanticParserPipeline import SemanticParserPipeline
from src.util.Storage import Storage


class PipelineTestSetCreator:
    def __init__(self,
                 dataset_difficulties: List[int] = None,
                 reranker_attached_configurations: List[bool] = None,
                 abstractor_configurations: List[AbstractionType] = None,
                 resolver_embedding_types: List[EmbeddingType] = None,
                 resolver_relative_not_sure_threshold: int = 2,
                 metric_learner_types: List[MetricLearnerType] = None,
                 bert_layer_types: List[BertKerasLayerType] = None,
                 table_bert_layer_types: List[BertKerasLayerType] = None,
                 table_embedder_pooling_types: List[TableEmbedderPoolingType] = None,
                 table_embedding_content_configurations: List[TableEmbeddingContent] = None,
                 reranker_lambda_embedder_attachments: List[LambdaEmbedderAttached] = None,
                 test_repititions: int = 20):
        self.models_dir = Path("../../res/trained/test_models/")
        self.current_subdir = ""
        self.data_storage = Storage()
        self.dataset_difficulties = dataset_difficulties
        self.reranker_attached_configurations = reranker_attached_configurations
        self.abstractor_configurations = abstractor_configurations
        self.resolver_embedding_types = resolver_embedding_types
        self.resolver_relative_not_sure_threshold = resolver_relative_not_sure_threshold
        self.metric_learner_types = metric_learner_types
        self.bert_layer_types = bert_layer_types
        self.table_bert_layer_types = table_bert_layer_types
        self.table_embedder_pooling_types = table_embedder_pooling_types
        self.table_embedding_content_configurations = table_embedding_content_configurations
        self.reranker_lambda_embedder_attachments = reranker_lambda_embedder_attachments
        self.test_repetitions = test_repititions

        self.resolve_inmutable_default_arguments(
            dataset_difficulties,
            reranker_attached_configurations,
            abstractor_configurations,
            resolver_embedding_types,
            metric_learner_types,
            bert_layer_types,
            table_bert_layer_types,
            table_embedder_pooling_types,
            table_embedding_content_configurations,
            reranker_lambda_embedder_attachments
        )

        self.datasets = self.load_required_datasets()

    def yield_pipeline_test_inputs(self) -> Iterable[Tuple[int, str, SemanticParserPipeline]]:
        argument_packs = [argument_pack for argument_pack in self.get_all_argument_packs()]
        test_pipeline_progress = tqdm(argument_packs)
        test_pipeline_progress.set_description("Running Tests")
        for argument_pack in test_pipeline_progress:
            yield argument_pack[0], argument_pack[1], self.create_semantic_parser_pipelines(*argument_pack)

    def get_all_argument_packs(self):
        for dataset_difficulty in self.dataset_difficulties:
            for abstractor_configuration in self.abstractor_configurations:
                for resolver_argument_pack in self.get_all_resolver_argument_packs():
                    for reranker_attached in self.reranker_attached_configurations:
                        all_reranker_argument_packs = [
                            reranker_argument_pack for reranker_argument_pack in self.get_all_reranker_argument_packs()
                        ]
                        if not reranker_attached:
                            reranker_argument_pack = all_reranker_argument_packs[0]
                            yield self.pack_inputs(
                                dataset_difficulty, abstractor_configuration, resolver_argument_pack, reranker_attached,
                                reranker_argument_pack
                            )
                        else:
                            for reranker_argument_pack in all_reranker_argument_packs:
                                yield self.pack_inputs(
                                    dataset_difficulty, abstractor_configuration, resolver_argument_pack,
                                    reranker_attached, reranker_argument_pack
                                )

    def create_semantic_parser_pipelines(
            self, dataset_difficulty: int, pipeline_name: str, abstractor_configuration: AbstractionType,
            reranker_attached: bool,
            resolver_argument_pack: Tuple[BertKerasLayerType, EmbeddingType, MetricLearnerType],
            reranker_argument_pack: Tuple[
                Tuple[BertKerasLayerType, TableEmbedderPoolingType, TableEmbeddingContent], LambdaEmbedderAttached
            ]) -> Iterable[SemanticParserPipeline]:

        embedding_function_provider = create_embedding_function_provider(resolver_argument_pack[0])
        save_location = self.resolve_save_directory_from(pipeline_name)
        resolver_dataset, condition_dataset, reranker_dataset = self.datasets[
            self.dataset_difficulties.index(dataset_difficulty)
        ]
        abstractor = create_abstractor(abstractor_configuration)
        for i in range(self.test_repetitions):
            candidate_resolver = create_resolver_model_from(
                resolver_dataset,
                "Lifted instance", "DSL output",
                embedding_function_provider,
                resolver_argument_pack[1],
                resolver_argument_pack[2],
                save_location / "candidate_resolver",
                relative_not_sure_threshold=self.resolver_relative_not_sure_threshold
            )
            condition_resolver = create_resolver_model_from(
                condition_dataset,
                "Lifted condition", "Lifted condition DSL",
                embedding_function_provider,
                resolver_argument_pack[1],
                resolver_argument_pack[2],
                save_location / "condition_resolver",
                take_best_guess=True,
                relative_not_sure_threshold=0
            )
            candidate_reranker = self.create_reranker_model_if_necessary(
                candidate_resolver, reranker_attached, reranker_dataset, reranker_argument_pack
            )
            yield SemanticParserPipeline(abstractor, candidate_resolver, condition_resolver, candidate_reranker)

    def get_all_resolver_argument_packs(self) -> Iterable[Tuple[BertKerasLayerType, EmbeddingType, MetricLearnerType]]:
        for bert_layer_type in self.bert_layer_types:
            for resolver_text_embedding_type in self.resolver_embedding_types:
                for metric_learner_type in self.metric_learner_types:
                    yield bert_layer_type, resolver_text_embedding_type, metric_learner_type

    def get_all_reranker_argument_packs(self) \
            -> Tuple[
                Tuple[BertKerasLayerType, TableEmbedderPoolingType, TableEmbeddingContent], LambdaEmbedderAttached
            ]:
        for bert_layer_type in self.table_bert_layer_types:
            for table_embedder_pooling_type in self.table_embedder_pooling_types:
                for table_embedding_contents in self.table_embedding_content_configurations:
                    for lambda_embedder_attached in self.reranker_lambda_embedder_attachments:
                        yield (
                            (bert_layer_type, table_embedder_pooling_type, table_embedding_contents),
                            lambda_embedder_attached
                        )

    def pack_inputs(self, dataset_difficulty: int, abstractor_configuration: AbstractionType,
                    resolver_argument_pack: Tuple[BertKerasLayerType, EmbeddingType, MetricLearnerType],
                    reranker_attached: bool,
                    reranker_argument_pack: Tuple[
                        Tuple[BertKerasLayerType, TableEmbedderPoolingType, TableEmbeddingContent],
                        LambdaEmbedderAttached
                    ]) -> Tuple:
        pipeline_name = self.get_semantic_parser_pipeline_name_from_arguments(
            dataset_difficulty, abstractor_configuration, reranker_attached,
            resolver_argument_pack, reranker_argument_pack
        )
        return (
            dataset_difficulty, pipeline_name, abstractor_configuration, reranker_attached,
            resolver_argument_pack, reranker_argument_pack
        )

    def resolve_save_directory_from(self, pipeline_name: str) -> Path:
        return self.models_dir / pipeline_name

    def create_reranker_model_if_necessary(
            self, candidate_resolver: CandidateResolver, reranker_attached: bool, reranker_dataset: pd.DataFrame,
            reranker_argument_pack: Tuple[
                Tuple[BertKerasLayerType, TableEmbedderPoolingType, TableEmbeddingContent], LambdaEmbedderAttached
            ]) -> CandidateReranker:
        candidate_reranker = None
        if reranker_attached:
            candidate_reranker = create_reranker_model_from(
                candidate_resolver,
                reranker_dataset,
                self.create_table_embedder(*reranker_argument_pack[0]),
                reranker_argument_pack[1]
            )
        return candidate_reranker

    @staticmethod
    def create_table_embedder(bert_layer_type: BertKerasLayerType,
                              table_embedder_pooling_type: TableEmbedderPoolingType,
                              table_embedding_contents: TableEmbeddingContent):
        return TableEmbedder.initialize(
            "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
            bert_layer_type.value,
            pooling_type=table_embedder_pooling_type.value,
            column_names_only=table_embedding_contents.value
        )

    @staticmethod
    def get_semantic_parser_pipeline_name_from_arguments(dataset_difficulty: int,
                                                         abstractor_configuration: AbstractionType,
                                                         reranker_attached: bool,
                                                         resolver_argument_pack: Tuple[
                                                             BertKerasLayerType, EmbeddingType, MetricLearnerType],
                                                         reranker_argument_pack: Tuple[
                                                             Tuple[
                                                                 BertKerasLayerType, TableEmbedderPoolingType,
                                                                 TableEmbeddingContent
                                                             ], LambdaEmbedderAttached
                                                         ]):
        return "abstractor{0}_resolver{1}{2}_difficulty{3}".format(
                abstractor_configuration.name,
                get_resolver_representation_from_arguments(resolver_argument_pack),
                get_reranker_representation_from_arguments(reranker_attached, reranker_argument_pack),
                dataset_difficulty
        )

    def load_required_datasets(self) -> List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        return [(
            self.data_storage.load_candidate_resolver_dataset(difficulty),
            self.data_storage.load_condition_input_data(self.data_storage.training_data_location),
            self.data_storage.load_reranking_dataset(difficulty)
        ) for difficulty in self.dataset_difficulties]

    def resolve_inmutable_default_arguments(self,
                                            dataset_difficulties: List[int],
                                            reranker_attached_configurations: List[bool],
                                            abstractor_configurations: List[AbstractionType],
                                            resolver_embedding_types: List[EmbeddingType],
                                            metric_learner_types: List[MetricLearnerType],
                                            bert_layer_types: List[BertKerasLayerType],
                                            table_bert_layer_types: List[BertKerasLayerType],
                                            table_embedder_pooling_types: List[TableEmbedderPoolingType],
                                            table_embedding_content_configurations: List[TableEmbeddingContent],
                                            reranker_lambda_embedder_attachments: List[LambdaEmbedderAttached]):
        if reranker_lambda_embedder_attachments is None:
            self.reranker_lambda_embedder_attachments = [LambdaEmbedderAttached.NO]
        if dataset_difficulties is None:
            self.dataset_difficulties = [1, 2, 3]
        if reranker_attached_configurations is None:
            self.reranker_attached_configurations = [True, False]
        if abstractor_configurations is None:
            self.abstractor_configurations = [AbstractionType.MOCK]
        if resolver_embedding_types is None:
            self.resolver_embedding_types = [
                EmbeddingType.SEQUENCE, EmbeddingType.BERT_POOLED, EmbeddingType.SEQUENCE_POSITIONAL,
                EmbeddingType.MAX_POOLED, EmbeddingType.AVG_POOLED, EmbeddingType.MAX_POOLED_POSITIONAL,
                EmbeddingType.AVG_POOLED_POSITIONAL
            ]
        if metric_learner_types is None:
            self.metric_learner_types = [metric_learner_type for metric_learner_type in MetricLearnerType]
        if bert_layer_types is None:
            self.bert_layer_types = [bert_layer_type for bert_layer_type in BertKerasLayerType]
        if table_bert_layer_types is None:
            self.table_bert_layer_types = self.bert_layer_types
        if table_embedder_pooling_types is None:
            self.table_embedder_pooling_types = [
                table_embedder_pooling_type for table_embedder_pooling_type in TableEmbedderPoolingType
            ]
        if table_embedding_content_configurations is None:
            self.table_embedding_content_configurations = [TableEmbeddingContent.COLUMN_ONLY]
