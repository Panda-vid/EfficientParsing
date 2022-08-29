import itertools
from functools import reduce
from typing import List, Tuple, Dict, Any

import numpy as np

from src.candidate_reranker.CandidateReranker import CandidateReranker
from src.candidate_resolver.CandidateResolver import CandidateResolver
from src.combination.FunctionTemplate import FunctionTemplate
from src.dataclasses.TableType import TableType
from src.entity_abstractor.Abstractor import Abstractor
from src.entity_abstractor.MockAbstractor import MockAbstractor
from src.util.Storage import Storage


# TODO: Have Enums for each configurable and define a function or class which configures objects of this class according to these configurables
# TODO: Have either a self-evaluation or external evaluation containing accuracy, recall and F1-measure, split measurements for each sub data set and normal data vs one-shot generalization
class SemanticParserPipeline:
    def __init__(self,
                 entity_abstractor: Abstractor | MockAbstractor,
                 candidate_resolver: CandidateResolver,
                 condition_resolver: CandidateResolver,
                 candidate_reranker: CandidateReranker = None,
                 active_context: str = None):
        self.entity_abstractor = entity_abstractor
        self.candidate_resolver = candidate_resolver
        self.condition_resolver = condition_resolver
        self.candidate_reranker = candidate_reranker
        self.active_context = active_context if active_context is not None else "meeting management"
        self.active_tables = Storage().load_context(self.active_context)

    def predict(self, utterance: str, active_context: str = None):
        self.update_active_context(active_context)
        abstractor_program_line_outputs = self.entity_abstractor.abstract(utterance, self.active_tables[0])
        output_probabilities, candidate_programs = self.generate_candidate_programs_and_assign_probability(
            abstractor_program_line_outputs
        )
        return self.select_predicted_program_from(
            output_probabilities, candidate_programs, abstractor_program_line_outputs, utterance
        )

    def retrain(self, original_utterance: str, decomposed_utterances: List[str], active_context: str = None) -> bool:
        self.update_active_context(active_context)
        composite_utterance_abstractor_output = self.entity_abstractor.abstract(
            original_utterance, self.active_tables[0]
        )[0]
        lifted_retraining_subprograms = [
            self.predict_retraining_subprogram(utterance, composite_utterance_abstractor_output)
            for utterance in decomposed_utterances
        ]
        return self.add_new_example_if_possible(
            original_utterance, lifted_retraining_subprograms,
            composite_utterance_abstractor_output[0]
        )

    def update_active_context(self, active_context: str):
        if active_context is not None:
            self.active_context = active_context
            self.active_tables = Storage().load_context(active_context)

    def generate_candidate_programs_and_assign_probability(
            self, abstractor_program_line_outputs: List[Tuple[str, Dict[str, Any], str]]) \
            -> Tuple[List[float], List[str]]:
        grounded_program_line_candidates = [
            self.get_grounded_program_line_candidates(lifted_subsentence, subprogram_inputs, lifted_condition)
            for lifted_subsentence, subprogram_inputs, lifted_condition in abstractor_program_line_outputs
        ]
        program_probability_tuples = [
            self.glue_grounded_subprograms_and_compute_probability(program_line_candidates)
            for program_line_candidates in itertools.product(*grounded_program_line_candidates)
        ]
        output_probabilities, candidate_programs = zip(*program_probability_tuples)
        return list(output_probabilities), list(candidate_programs)

    def predict_retraining_subprogram(self,
                                      decomposition_utterance: str,
                                      composite_utterance_abstractor_output: Tuple[str, Dict[str, Any], str]):
        abstractor_outputs = self.entity_abstractor.abstract(
            decomposition_utterance, self.active_tables[0]
        )[0]
        abstractor_outputs_containing_constant_inputs = \
            self.get_abstractor_outputs_containing_constant_inputs_for_composite_template(
                abstractor_outputs, composite_utterance_abstractor_output
            )
        output_probabilities, candidate_subprograms = self.generate_candidate_programs_and_assign_probability(
            abstractor_outputs_containing_constant_inputs
        )
        return self.select_predicted_program_from(
            output_probabilities, candidate_subprograms,
            abstractor_outputs_containing_constant_inputs, decomposition_utterance
        )

    def select_predicted_program_from(self,
                                      output_probabilities: List[float],
                                      candidate_programs: List[str],
                                      abstractor_program_line_outputs: List[Tuple[str, Dict[str, Any], str]],
                                      utterance: str) -> str:
        if len(candidate_programs) > 0 and self.candidate_reranker is not None:
            predicted_program = self.use_candidate_reranker(
                output_probabilities,
                candidate_programs,
                abstractor_program_line_outputs,
                utterance
            )
        elif len(candidate_programs) > 0:
            predicted_program = candidate_programs[0]
        else:
            predicted_program = "NOT_SURE"
        return predicted_program

    def add_new_example_if_possible(self, utterance: str, lifted_retraining_subprograms: List[str], lifted_composite_utterance: str):
        success = "NOT_SURE" not in lifted_retraining_subprograms
        if success:
            lifted_retraining_program = ";".join(lifted_retraining_subprograms)
            self.add_training_instance_and_retrain(utterance, lifted_composite_utterance, lifted_retraining_program)
        return success

    @staticmethod
    def get_abstractor_outputs_containing_constant_inputs_for_composite_template(
            abstractor_outputs: List[Tuple[str, Dict[str, Any], str]],
            composite_utterance_abstractor_output: Tuple[str, Dict[str, Any], str]):
        for index, lifted_atomic_utterance, atomic_action_inputs, atomic_action_condition \
                in enumerate(abstractor_outputs):
            filtered_input_dict = {}
            for table_type, typed_inputs in atomic_action_inputs:
                composite_typed_inputs = composite_utterance_abstractor_output[1][table_type]
                filtered_inputs = filter(
                    lambda typed_input: typed_input not in composite_typed_inputs, typed_inputs
                )
                filtered_input_dict[table_type] = filtered_inputs
            abstractor_outputs[index][1] = filtered_input_dict
        return abstractor_outputs

    def get_grounded_program_line_candidates(self,
                                             lifted_subsentence: str,
                                             subprogram_inputs: Dict[str, Any],
                                             lifted_condition: str) -> List[Tuple[float, str]]:
        output_probabilities, candidate_program_lines = self.candidate_resolver.call(lifted_subsentence)
        condition_probability, program_condition_candidate = self.condition_resolver.call(lifted_condition)
        return [
            (
                output_probability,
                FunctionTemplate.ground_lifted_program(
                    candidate_program_line, subprogram_inputs, program_condition_candidate
                )
            )
            for output_probability, candidate_program_line in zip(output_probabilities, candidate_program_lines)
        ]

    def add_training_instance_and_retrain(self, utterance: str, lifted_utterance: str, lifted_program: str):
        self.candidate_resolver.add_training_example_and_retrain(lifted_utterance, lifted_program)
        if self.candidate_reranker is not None:
            self.candidate_reranker.retrain(utterance, self.active_tables, lifted_program)

    @staticmethod
    def glue_grounded_subprograms_and_compute_probability(program_line_candidates: Tuple[Tuple[float, str], ...])\
            -> Tuple[float, str]:
        program_line_candidates = list(program_line_candidates)
        output_probabilities = np.array([
            program_line_candidate[0] for program_line_candidate in program_line_candidates
        ], dtype=np.float)
        program_lines = [program_line_candidate[1] for program_line_candidate in program_line_candidates]
        return np.mean(output_probabilities), ";\n".join(program_lines)

    def use_candidate_reranker(self,
                               output_probabilities: List[float],
                               candidate_programs: List[str],
                               abstractor_program_line_outputs: List[Tuple[str, Dict[str, Any], str]],
                               utterance: str) -> str:
        input_tables = self.get_input_tables_from(abstractor_program_line_outputs)
        predicted_program = self.candidate_reranker.select_best_candidate(
            utterance, output_probabilities, candidate_programs, input_tables
        )
        return predicted_program

    @staticmethod
    def get_input_tables_from(abstractor_program_line_outputs: List[Tuple[str, Dict[str, Any], str]]):
        return [
            table for abstractor_program_line_output in abstractor_program_line_outputs
            for table in reduce(
                lambda table_pack1, table_pack2: table_pack1 + table_pack2,
                abstractor_program_line_output[1][TableType.TABLE]
            )
        ]

    def save(self):
        # TODO: save model pipeline
        pass

    def load(self):
        # TODO: load model pipeline
        pass
