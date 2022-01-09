import unittest

from smart_ccg.sem_parser.entity_abstractor.Abstractor import Abstractor


class AbstractorTest(unittest.TestCase):
    def setUp(cls) -> None:
        cls.abstractor = Abstractor()

    def test_abstractor_initialization(self):
        parse, = self.abstractor.dependency_parser.raw_parse("This is a test.")
        self.assertEqual("This\tDT\t4\tnsubj\nis\tVBZ\t4\tcop\na\tDT\t4\tdet\ntest\tNN\t0\tROOT\n.\t.\t4\tpunct\n",
                          parse.to_conll(4))

    def test_abstractor_sentenece_with_case(self):
        sentence, subsentences, lifted_string = self.abstractor.lifted_sentence_instances(
            "Sort them by age ascending."
        )
        self.assertEqual("Sort <obj> <case>.", lifted_string)

    def test_abstractor_sentence_with_object_list(self):
        sentence, subsentences, lifted_string = self.abstractor.lifted_sentence_instances(
            "Show me the colors, age and range of all the vehicles."
        )
        self.assertEqual("Show me <objs>[4, 6, 8] <case> <obj>.", lifted_string)

    def test_abstractor_lifted_composed_sentence(self):
        sentence, subsentences, lifted_string = self.abstractor.lifted_sentence_instances(
            "Show me the colors, age and range of the vehicles with a range larger than 300 miles and an age of less than 2 years and sort them by age ascending.")
        self.assertEqual("Show me <objs>[4, 6, 8] <case> <obj> and sort <obj> <case>.", lifted_string)


if __name__ == '__main__':
    unittest.main()
