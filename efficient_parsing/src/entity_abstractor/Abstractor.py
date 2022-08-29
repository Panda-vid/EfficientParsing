from typing import List, Tuple, Dict, Any

from nltk import CoreNLPDependencyParser

from src.datamodel.Table import Table
from src.entity_abstractor.dependencytree.creation.LiftableDependencyTreeCreator import LiftableDependencyTreeCreator
from src.entity_abstractor.dependencytree.creation.LiftableDependencyTreeNodeFactory import \
    LiftableDependencyTreeNodeFactory
from src.datamodel.Sentence import Sentence


class Abstractor:
    def __init__(self):
        self.dependency_parser = CoreNLPDependencyParser()

    def abstract(self, utterance: str, table: Table) -> List[Tuple[str, Dict[str, Any], str]]:
        sentence, subsentences = self.extract_sentence_instances_from(utterance)
        return [sentence.abstract(table)] if len(subsentences) == 0 \
            else [subsentence.abstract(table) for subsentence in subsentences]

    def extract_sentence_instances_from(self, utterance: str) -> Tuple[Sentence, List[Sentence]]:
        raw_nodes = self.extract_raw_dependency_parse_nodes_from(utterance)
        dependency_tree_creator = LiftableDependencyTreeCreator(raw_nodes, LiftableDependencyTreeNodeFactory
                                                                .get_default_instance())
        sentence_tree = dependency_tree_creator.create_tree()

        return (
            Sentence.create_sentence(sentence_tree),
            [
                Sentence.create_sentence(subsentence_tree)
                for subsentence_tree in dependency_tree_creator.create_subsentence_trees_of(sentence_tree)
            ]
        )

    def extract_raw_dependency_parse_nodes_from(self, utterance: str) -> List[Tuple[str, str, int, str]]:
        parse, = self.dependency_parser.raw_parse(utterance)
        res = []
        for pos, conll_line in enumerate(parse.to_conll(4).split("\n")[:-1]):
            word, word_type, parent_id, dependency = tuple(conll_line.split())
            res.append((word, word_type, int(parent_id), dependency))
        return res
