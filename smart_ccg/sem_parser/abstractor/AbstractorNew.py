from typing import List, Tuple

from nltk import CoreNLPDependencyParser

from smart_ccg.sem_parser.abstractor.sentence.SentenceNew import Sentence
from smart_ccg.sem_parser.abstractor.dependencytree.creation.LiftableDependencyTreeCreator import \
    LiftableDependencyTreeCreator
from smart_ccg.sem_parser.abstractor.dependencytree.creation.LiftableDependencyTreeNodeFactory import \
    LiftableDependencyTreeNodeFactory


class Abstractor:
    def __init__(self):
        self.dependency_parser = CoreNLPDependencyParser()

    def extract_sentence_instances_from(self, sentence: str) -> Tuple[Sentence, List[Sentence]]:
        raw_nodes = self.extract_raw_nodes_from(sentence)
        dependency_tree_creator = LiftableDependencyTreeCreator(raw_nodes,
                                                                LiftableDependencyTreeNodeFactory.get_default_instance())
        sentence_tree = dependency_tree_creator.create_tree()
        return Sentence.create_sentence(sentence_tree), [Sentence.create_sentence(subsentence_tree) for subsentence_tree
                                                         in dependency_tree_creator
                                                             .create_subsentence_trees_of(sentence_tree)]

    def extract_raw_nodes_from(self, sentence: str) -> List[Tuple[str, str, int, str]]:
        parse, = self.dependency_parser.raw_parse(sentence)
        res = []
        for pos, conll_line in enumerate(parse.to_conll(4).split("\n")[:-1]):
            print(pos + 1, conll_line)
            word, word_type, parent_id, dependency = tuple(conll_line.split())
            res.append((word, word_type, int(parent_id), dependency))
        return res


if __name__ == '__main__':
    sentence_string = "Select all students that have passed the exam."
    abstractor = Abstractor()
    sentence, subsentences = abstractor.extract_sentence_instances_from(sentence_string)
    print(sentence.values, sentence.objects, sentence.cases)
    print(sentence.lifted())
    print(sentence.case_lifted())
    for subsentence in subsentences:
        print(subsentence.lifted())
