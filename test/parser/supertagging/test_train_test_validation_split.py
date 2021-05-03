import unittest

from smart_ccg.parser.supertagging.train_test_validation_split import load_annotated_sentences_from_processed_file,\
    train_test_validation_split
from pathlib import Path


class TrainTestValidationSplitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        filepath = Path("/home/insert/PycharmProjects/SmartCCG/resources/supertagging/gmb_processed/gmb_processed")
        cls.sentences = load_annotated_sentences_from_processed_file(filepath)

    def test_number_of_entries_per_set(self):
        test_size = 0.3
        val_size = 0.2

        num_sentences = len(self.sentences)

        train, test, validation = train_test_validation_split(self.sentences, test_size, val_size)
        num_train, num_test, num_validation = len(train), len(test), len(validation)

        self.assertEqual(num_train + num_test + num_validation, num_sentences)
        self.assertEqual(num_test, int(test_size * num_sentences) + 1)
        self.assertEqual(num_validation, int(val_size * num_sentences) + 1)
        self.assertEqual(num_train, int((1 - test_size - val_size) * num_sentences))

    def test_test_size_larger_one(self):
        test_size = 1.2
        val_size = 0.2

        self.assertRaises(ValueError, train_test_validation_split, self.sentences, test_size, val_size)

    def test_val_size_larger_one(self):
        test_size = 0.2
        val_size = 1.2

        self.assertRaises(ValueError, train_test_validation_split, self.sentences, test_size, val_size)

    def test_test_size_negative(self):
        test_size = -0.2
        val_size = 0.2

        self.assertRaises(ValueError, train_test_validation_split, self.sentences, test_size, val_size)

    def test_val_size_negative(self):
        test_size = 0.2
        val_size = -0.2

        self.assertRaises(ValueError, train_test_validation_split, self.sentences, test_size, val_size)

    def test_val_size_larger_than_train_size(self):
        test_size = 0.3
        val_size = 0.71

        self.assertRaises(ValueError, train_test_validation_split, self.sentences, test_size, val_size)


if __name__ == '__main__':
    unittest.main()
