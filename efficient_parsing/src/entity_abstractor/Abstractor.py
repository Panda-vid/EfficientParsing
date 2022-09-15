import requests
import subprocess
import shlex
import time

from pathlib import Path
from typing import List, Tuple, Dict, Any

from nltk import CoreNLPDependencyParser
from requests.exceptions import ConnectionError

from src.datamodel.Sentence import Sentence
from src.datamodel.Table import Table
from src.entity_abstractor.dependencytree.creation.LiftableDependencyTreeCreator import LiftableDependencyTreeCreator
from src.entity_abstractor.dependencytree.creation.LiftableDependencyTreeNodeFactory import \
    LiftableDependencyTreeNodeFactory


class Abstractor:
    def __init__(self):
        self.dependency_parser = CoreNLPDependencyParser()

    @staticmethod
    def start_core_nlp_server():
        server_path = Path(__file__).parents[3] / "res" / "abstraction" / "stanford-corenlp-4.2.2"
        command = "java -mx4g -cp '*' edu.stanford.nlp.pipeline.StanfordCoreNLPServer " + \
                  "-preload tokenize,ssplit,pos,lemma,ner,parse,depparse -status_port 9000 -port 9000 -timeout 15000 &"
        subprocess.Popen(
            shlex.split(command),
            stdout=subprocess.PIPE, cwd=server_path
        )

    @staticmethod
    def wait_until_server_reachable():
        response_code = 0
        while response_code != 200:
            time.sleep(5)
            try:
                response_code = requests.post(
                    'http://[::]:9000/?properties={"annotators":"tokenize,ssplit,pos","outputFormat":"json"}',
                    data={'data': 'The quick brown fox jumped over the lazy dog.'}
                ).status_code
            except ConnectionError as error:
                pass

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

    @staticmethod
    def stop_core_nlp_server():
        url = "http://localhost:9000/shutdown?"
        shutdown_key = subprocess.getoutput("cat /tmp/corenlp.shutdown")
        requests.post(url, data="", params={"key": shutdown_key})
