import re
from typing import Iterator, List

from more_itertools import flatten
from pattern.text.en import singularize


def normalize(words: str) -> str:
    """
    Normalizes camel case or snake case or combined camel case and snake case words containing plurals to a single space
    separated string.
    :param words:
    :return:
    """
    split_words = split(words)
    singular_words = all_singular(split_words)
    return " ".join(all_lower(singular_words))


def split(words: str) -> Iterator[str]:
    res = split_snake_case(words)
    return flatten(split_camel_case(string) for string in res)


def split_snake_case(words: str) -> List[str]:
    return words.split(sep="_")


def split_camel_case(words: str) -> Iterator[str]:
    lower_split = split_lower_camel_case_and_leading_acronyms(words)
    return flatten(split_capital_numbers_and_trailing_acronyms_in_camel_case(string) for string in lower_split)


def split_lower_camel_case_and_leading_acronyms(words: str) -> List[str]:
    """
    Splits lower case camel case words and leading acronyms.
    e.g. "aVariable" -> ["a", "variable"] or HTTPResponse -> ["HTTP", "Response"]
    :param words:
    :return:
    """
    pattern = re.compile('(.)([A-Z][a-z]+)')
    return re.split(pattern, words)


def split_capital_numbers_and_trailing_acronyms_in_camel_case(words: str) -> List[str]:
    """
    Splits capitalized camel case strings which have numbers in them or lower case camel case followed by an acronym.
    e.g "This1Class" -> ["This1", "Class"] or "getHTTP" -> ["get", "HTTP"]
    :param words:
    :return:
    """
    pattern = re.compile(r'([a-z\d])([A-Z])')
    return re.split(pattern, words)


def all_singular(words: Iterator[str]) -> List[str]:
    return [singular(word) for word in words]


def all_lower(words: Iterator[str]) -> List[str]:
    return [word.lower() for word in words]


def singular(word: str) -> str:
    return singularize(word)


def remove_punctuation(word: str) -> str:
    return word.translate(str.maketrans("", "", ".,;!?"))


def make_capital_snake_case(words: List[str]) -> str:
    return make_snake_case([words[0].capitalize()] + words[1:])


def make_all_capital_snake_case(words: List[str]) -> str:
    return make_snake_case([word.capitalize() for word in words])


def make_upper_snake_case(words: List[str]) -> str:
    return make_snake_case([word.upper() for word in words])


def make_snake_case(words: List[str]) -> str:
    return "_".join(words)


def make_camel_case(words: List[str]) -> str:
    return "".join([words[0].lower(), make_capital_camel_case(words[1:])])


def make_capital_camel_case(words: List[str]) -> str:
    return "".join([word.capitalize() for word in words])


def make_compound(words: List[str]) -> str:
    return "".join(words)


def make_space_separated_compound(words: List[str]) -> str:
    return " ".join(words)
