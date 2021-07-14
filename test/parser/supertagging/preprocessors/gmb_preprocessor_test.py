import shutil
import unittest

from smart_ccg.sem_parser.ccg_parser.supertagging.preprocessors.gmb_preprocessor import process_gmb_files
from test.test_utils import get_test_resource_directory


class GMBPreprocessorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        test_resource_directory = get_test_resource_directory()
        cls.test_gmb_directory = test_resource_directory / "supertagging" / "gmb_subset_testing"
        cls.test_gmb_processed_directory = test_resource_directory / "tmp" / "gmb_test_processing"
        cls.test_gmb_processed_directory.mkdir(parents=True)

        process_gmb_files(cls.test_gmb_directory, cls.test_gmb_processed_directory)

    def test_forward_separation_of_sentences(self):
        processed_file = self.test_gmb_processed_directory / "gmb_processed"
        self.assertTrue(processed_file.is_file())
        eos_symbols = [".", "?", "!", '"']

        with processed_file.open("r") as f:
            lines = f.readlines()
            for i in range(len(lines)):
                if lines[i] == "\n" and i >= 1:

                    tokens = lines[i-1].split("\t")
                    previous_is_eos_symbol = tokens[0] in eos_symbols
                    previous_is_noun = tokens[1].strip("\n") == "N"
                    self.assertTrue(previous_is_eos_symbol or previous_is_noun)

    def test_separation_of_sentences_after_eos_symbols(self):
        processed_file = self.test_gmb_processed_directory / "gmb_processed"
        self.assertTrue(processed_file.is_file())
        eos_symbols = [".", "?", "!"]

        with processed_file.open("r") as f:
            lines = f.readlines()

            for i in range(len(lines)):
                if lines[i] in eos_symbols and i < len(lines):
                    self.assertEquals(lines[i+1], "\n")

    @classmethod
    def tearDownClass(cls):
        tmp_directory = cls.test_gmb_processed_directory.parent
        shutil.rmtree(tmp_directory)


if __name__ == '__main__':
    unittest.main()
