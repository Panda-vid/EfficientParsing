import unittest

import pandas as pd

from smart_ccg.sem_parser.abstractor.Abstractor import Abstractor
from smart_ccg.sem_parser.abstractor.Table import Table


class AbstractorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.abstractor = Abstractor()

    def test_abstractor_initialization(self):
        parse, = self.abstractor.dependency_parser.raw_parse("This is a test.")
        self.assertEqual("This\tDT\t4\tnsubj\nis\tVBZ\t4\tcop\na\tDT\t4\tdet\ntest\tNN\t0\tROOT\n.\t.\t4\tpunct\n",
                         parse.to_conll(4))

    def test_abstractor_no_table_single_sentence_object_list(self):
        string = "Create the table students containing the columns name, surname, mark and id."
        sentence, subsentences = self.abstractor.extract_sentence_instances_from(string)

        self.assertEqual(0, len(sentence.values))
        self.assertEqual(["students", "name", "surname", "mark", "id"], [obj.word for obj in sentence.objects])
        self.assertEqual(0, len(sentence.cases))
        self.assertEqual("Create [table] [,column]", sentence.lifted())
        self.assertEqual(0, len(subsentences))

    def test_abstractor_no_table_single_sentence_object_list_ungrammatical(self):
        string = "Create the table students containing the columns name surname mark and id."
        sentence, subsentences = self.abstractor.extract_sentence_instances_from(string)

        self.assertEqual(0, len(subsentences))
        self.assertNotEqual("Create [table] [,column]", sentence.lifted())

    def test_abstractor_table_single_sentence_condition(self):
        string = "Show all students which have passed the exam."
        table = Table(["exam", "id"], "students", pd.DataFrame())
        sentence, subsentences = self.abstractor.extract_sentence_instances_from(string)

        self.assertEqual(0, len(subsentences))
        self.assertEqual(["passed"], [value.word for value in sentence.values])
        self.assertEqual(["students", "exam"], [obj.word for obj in sentence.objects])
        self.assertEqual(["have"], [case.word for case in sentence.cases])
        self.assertEqual("Show [table] [case]", sentence.lifted(table))
        self.assertEqual(["have [value] [column]"], sentence.case_lifted(table))


if __name__ == '__main__':
    unittest.main()
