import pandas as pd

from src.candidate_reranker.lambda_calculus_embedding.depccg_interface import get_lambda_of, setup_model
from src.candidate_reranker.lambda_calculus_embedding.lambda_embedding_function_providers \
    import create_lambda_embedder_function


class LambdaEmbedder:
    def __init__(
            self,
            dataset,
            label_column_name,
            lambda_embedder_function,
            lambda_annotator_function,
            no_hit_included):
        self.dataset = dataset
        self.label_column_name = label_column_name
        self.lambda_annotator_function = lambda_annotator_function
        self.lambda_embedder_function = lambda_embedder_function
        self.no_hit_included = no_hit_included

    def __call__(self, utterance: str, print_lambda_calculus: bool = False):
        lambda_calc = self.lambda_annotator_function(utterance)
        if print_lambda_calculus:
            print(lambda_calc)
        return self.lambda_embedder_function(lambda_calc)

    def retrain(self, lifted_candidate_program: str):
        self.append_new_label_to_dataset(lifted_candidate_program)
        self.lambda_embedder_function = create_lambda_embedder_function(
            self.dataset, self.label_column_name, self.no_hit_included
        )

    def append_new_label_to_dataset(self, new_label: str):
        new_line = pd.DataFrame({self.label_column_name: new_label})
        self.dataset = pd.concat([self.dataset, new_line])


class Builder:
    def __init__(self):
        self.lambda_embedder_dataset = None
        self.label_column_name = None
        self.lambda_embedder_include_no_hit = False
        self.depccg_annotator = "spacy"
        self.depccg_language = "en"
        self.depccg_unary_penalty = 0.1
        self.depccg_nbest = 1
        self.depccg_pruning_size = 50
        self.depccg_beta = 1e-4
        self.depccg_use_beta = True
        self.depccg_max_length = 250
        self.depccg_max_step = int(1e7)
        self.depccg_num_processes = 4

    def set_lambda_embedder_data(self, dataset: pd.DataFrame, label_column_name: str):
        self.lambda_embedder_dataset = dataset
        self.label_column_name = label_column_name
        return self

    def lambda_embedder_include_no_hit_in_encoding(self, include_no_hit: bool):
        self.lambda_embedder_include_no_hit = include_no_hit
        return self

    def set_depccg_annotator(self, depccg_annotator: str):
        self.depccg_annotator = depccg_annotator
        return self

    def set_depccg_language(self, language: str):
        self.depccg_language = language
        return self

    def set_depccg_unary_penalty(self, unary_penalty: float):
        self.depccg_unary_penalty = unary_penalty
        return self

    def set_depccg_nbest(self, nbest: int):
        self.depccg_nbest = nbest
        return self

    def set_depccg_pruning_size(self, pruning_size: int):
        self.depccg_pruning_size = pruning_size
        return self

    def set_depccg_beta(self, beta: float):
        self.depccg_beta = beta
        self.depccg_use_beta = True
        return self

    def use_depccg_beta(self, use_beta: bool):
        self.depccg_use_beta = use_beta
        return self

    def set_depccg_max_length(self, max_length: int):
        self.depccg_max_length = max_length
        return self

    def set_depccg_max_step(self, max_step: int):
        self.depccg_max_step = max_step
        return self

    def set_depccg_num_processes(self, num_processes: int):
        self.depccg_num_processes = num_processes
        return self

    def build(self):
        if not self.all_set():
            raise ValueError(
                "Not all attributes of the LambdaEmbedder class have been set.\n" +
                "Check that you have called 'LambdaEmbedderBuilder.set_lambda_embedder_data()' before building."
            )
        depccg_supertagger, depccg_config, depccg_annotator_function, depccg_annotator, depccg_kwargs = setup_model(
            annotator=self.depccg_annotator, language=self.depccg_language, unary_penalty=self.depccg_unary_penalty,
            nbest=self.depccg_nbest, pruning_size=self.depccg_pruning_size, beta=self.depccg_beta,
            use_beta=self.depccg_use_beta, max_length=self.depccg_max_length, max_step=self.depccg_max_step,
            num_processes=self.depccg_num_processes
        )

        def lambda_annotator_function(input_str: str):
            return get_lambda_of(
                input_str, depccg_supertagger, depccg_config,
                depccg_annotator_function, depccg_annotator, depccg_kwargs
            )

        lambda_embedder_function = create_lambda_embedder_function(
            self.lambda_embedder_dataset, self.label_column_name, self.lambda_embedder_include_no_hit
        )
        return LambdaEmbedder(
            self.lambda_embedder_dataset,
            self.label_column_name,
            lambda_embedder_function,
            lambda_annotator_function,
            self.lambda_embedder_include_no_hit
        )

    def all_set(self) -> bool:
        return all(value is not None for key, value in self.__dict__.items())
