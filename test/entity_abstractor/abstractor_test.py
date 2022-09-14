import unittest

import pandas as pd

from src.datamodel.Table import Table
from src.entity_abstractor.Abstractor import Abstractor


class AbstractorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.abstractor = Abstractor()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.abstractor.stop_core_nlp_server()

    def test_abstractor_initialization(self):
        parse, = self.abstractor.dependency_parser.raw_parse("This is a test.")
        self.assertEqual("This\tDT\t4\tnsubj\nis\tVBZ\t4\tcop\na\tDT\t4\tdet\ntest\tNN\t0\tROOT\n.\t.\t4\tpunct\n",
                         parse.to_conll(4))

    def test_abstractor_no_table_single_sentence_object_list(self):
        string = "Create the table students containing the columns name, surname, mark and id."
        sentence, subsentences = self.abstractor.extract_sentence_instances_from(string)

        self.assertEqual(0, len(sentence.values))
        self.assertEqual(["students", "name", "surname", "mark", "id"],
                         [obj.word for pack in sentence.objects for obj in pack])
        self.assertEqual(0, len(sentence.cases))
        self.assertEqual("Create [table] [,column]", sentence.lifted())
        self.assertEqual(1, len(subsentences))

    def test_abstractor_no_table_single_sentence_object_list_ungrammatical(self):
        string = "Create the table students containing the columns name surname mark and id."
        sentence, subsentences = self.abstractor.extract_sentence_instances_from(string)

        self.assertEqual(1, len(subsentences))
        self.assertEqual("Create [table] [,column]", sentence.lifted())
        self.assertNotEqual(["students", "name", "surname", "mark", "id"],
                         [obj.word for pack in sentence.objects for obj in pack])

    def test_abstractor_table_single_sentence_condition(self):
        string = "Show all students which have passed the exam."
        table = Table.create_test_table_instance(["exam", "id"], "students", pd.DataFrame())
        sentence, subsentences = self.abstractor.extract_sentence_instances_from(string)

        self.assertEqual(1, len(subsentences))
        self.assertEqual(["passed"], [value.word for value in sentence.values])
        self.assertEqual(["students", "exam"], [obj.word for pack in sentence.objects for obj in pack])
        self.assertEqual(["have"], [case.word for case in sentence.cases])
        self.assertEqual("Show [table] [condition]", sentence.lifted(table))
        self.assertEqual("have [value] [column]", sentence.case_lifted(table))

    def test_abstractor_table_no_condition(self):
        string = "From students select the exam and id."
        table = Table.create_test_table_instance(["exam", "id"], "students", pd.DataFrame())
        sentence, subsentences = self.abstractor.extract_sentence_instances_from(string)
        self.assertEqual(1, len(subsentences))
        self.assertEqual(["students", "exam", "id"], [obj.word for pack in sentence.objects for obj in pack])
        print(sentence.case_lifted(table))
        self.assertEqual("[table] select [,column]", sentence.lifted(table))


if __name__ == '__main__':
    unittest.main()
