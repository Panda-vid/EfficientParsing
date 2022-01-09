import subprocess
import time
from typing import Tuple, List

from nltk.parse.corenlp import CoreNLPDependencyParser

from smart_ccg.sem_parser.entity_abstractor.DependencyTreeLifter import DependencyTreeLifter
from smart_ccg.sem_parser.entity_abstractor.Sentence import Sentence


class Abstractor:
    def __init__(self):
        # TODO: cwd is hardcoded
        # starts the CoreNLPServer
        self.server_process = subprocess.Popen(['java', '-mx4g', '-cp', '*',
                                                'edu.stanford.nlp.pipeline.StanfordCoreNLPServer', '-preload',
                                                'tokenize,ssplit,pos,lemma,ner,parse,depparse', '-status_port', '9000',
                                                '-port', '9000', '-timeout', '15000', '&'],
                                               cwd="/home/pandavid/PycharmProjects/SmartCCG/resources/abstractor"
                                                   + "/stanford-corenlp-4.2.2/")
        time.sleep(30)
        self.dependency_parser = CoreNLPDependencyParser()
        self.dependency_tree_lifter = DependencyTreeLifter.get_instance()
        self.object_dependency_tokens = ["obj", "appos", "nmod"]
        self.oblique_dependency_tokens = ["obl"]
        self.conjunction_dependency_tokens = ["conj"]

    def lifted_sentence_instances(self, string: str) -> Tuple[Sentence, List[Sentence], str]:
        sentence, subsentences = Sentence.create_sentnece_instances_from(string, self.dependency_parser,
                                                                         self.object_dependency_tokens,
                                                                         self.oblique_dependency_tokens,
                                                                         self.conjunction_dependency_tokens)
        print(sentence.as_conll())
        return sentence, subsentences, sentence.get_lifted_string()

    def lifted_instances(self, sentence: str):
        raw_nodes = self.create_raw_node_list(sentence)
        sentence = Sentence.create_sentnece_instance_from(raw_nodes)

    def create_raw_node_list(self, sentence: str) -> List[Tuple[str, str, int, str]]:
        parse, = self.dependency_parser.raw_parse(sentence)
        res = []
        for conll_line in parse.to_conll(4).split("\n")[:-1]:
            word, word_type, parent_id, dependency = tuple(conll_line.split())
            res.append((word, word_type, int(parent_id), dependency))
        return res

    def __del__(self):
        self.server_process.terminate()
        del self.server_process

