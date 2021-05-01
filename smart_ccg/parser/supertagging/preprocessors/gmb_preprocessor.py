from pathlib import Path
from tqdm import tqdm


def process_gmb_files(gmb_directory, target_directory):
    if not target_directory.exists():
        target_directory.mkdir(parents=True)

    if not gmb_directory.exists():
        raise FileNotFoundError("Could not find the GMB root directory under: []"
                                .format(gmb_directory.absolute()))

    for document in tqdm(gmb_directory.rglob("data/p*/d*"),
                         desc="Number of documents processed", total=10000):
        target_file = target_directory / "gmb_processed"
        token_file = document / "en.tags"
        if token_file.is_file():
            process_document_token_file_and_save(token_file, target_file)


def process_document_token_file_and_save(input_token_file, target_file):
    eos_symbols = [".", "?", "!", '"']

    with input_token_file.open("r") as input_file, target_file.open("a") as target:
        lines = input_file.readlines()
        for i in range(len(lines)):
            if lines[i] != "\n" and lines[i] != "":
                tokens = lines[i].split("\t")
                target.write("{}\t{}\n".format(tokens[0], tokens[8]))
            elif lines[i] == "\n" and i >= 1:
                previous_tokens = lines[i-1].split()
                previous_is_eos_symbol = previous_tokens[0] in eos_symbols
                previous_is_noun = previous_tokens[8].strip("\n") == "N"
                if previous_is_noun or previous_is_eos_symbol:
                    target.write("\n")
        target.write("\n")


if __name__ == "__main__":
    gmb_path = Path("/home/pandavid/Downloads/gmb-2.2.0/")
    target_path = Path("/home/pandavid/PycharmProjects/SmartCCG/resources/supertagging/gmb_processed")
    process_gmb_files(gmb_path, target_path)
