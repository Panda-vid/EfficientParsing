from pathlib import Path
from sklearn.model_selection import train_test_split


def split_processed_dataset_and_save(filepath_processed_dataset, target_directory, test_size=0.2, val_size=0.05):
    sentences = load_annotated_sentences_from_processed_file(filepath_processed_dataset)

    train, test, validation = train_test_validation_split(sentences, test_size, val_size)

    if not target_directory.exists():
        target_directory.mkdir(parents=True)

    train_filepath = target_directory / "processed_train"
    test_filepath = target_directory / "processed_test"
    validation_filepath = target_directory / "processed_validation"

    with train_filepath.open("w") as train_file, test_filepath.open("w") as test_file, \
            validation_filepath.open("w") as validation_file:
        train_file.write("\n\n".join(train))
        test_file.write("\n\n".join(test))
        validation_file.write("\n\n".join(validation))


def train_test_validation_split(sentences, test_size, val_size):
    if test_size > 1 or val_size > 1:
        raise ValueError("test_size and/or val_size cannot be larger than 1:\ngiven: test_size={}, val_size={}"
                         .format(test_size, val_size))

    if val_size < 0 or test_size < 0:
        raise ValueError("test_size and/or val_size cannot be negative:\ngiven: test_size={}, val_size={}"
                         .format(test_size, val_size))

    train_size = 1 - test_size

    if val_size > train_size:
        raise ValueError("val_size cannot be larger than the size of the training set which is '1 - test_size':\n" +
                         "given: test_size={}, train_size={}, val_size={}".format(test_size, train_size, val_size))


    val_size = int(val_size * len(sentences)) + 1
    train, test = train_test_split(sentences, test_size=test_size, shuffle=True)
    train, val, = train_test_split(train, test_size=val_size, shuffle=True)

    return train, test, val


def load_annotated_sentences_from_processed_file(filepath_processed_dataset):
    if filepath_processed_dataset.is_file():
        with filepath_processed_dataset.open("r") as processed:
            content = processed.read()
            sentences = content.split("\n\n")
    else:
        raise FileNotFoundError("Could not find the processed dataset file under: {}"
                                .format(filepath_processed_dataset.absolute()))

    return sentences


if __name__ == '__main__':
    filepath = Path("/home/insert/PycharmProjects/SmartCCG/resources/supertagging/gmb_processed/gmb_processed")
    target = filepath.parent
    split_processed_dataset_and_save(filepath, target)
