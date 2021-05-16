import re

from Action import Action
from Input import Input, Type
from Conjunction import Conjunction


def parse_grammar(file_location):
    actions = []
    subactions = []
    conjunctions = []
    conditions = []
    with file_location.open("r") as grammar_file:
        in_category = False
        comment = False
        for line in grammar_file:
            if line == '"""':
                comment = not comment
            if not comment:
                match = re.search("<(.+?)>", line)
                if match:
                    in_category, category = extract_category(match, in_category)
                else:
                    line = line.strip(" ", "\t")
                    if category == "complete":
                        actions.append(parse_action(line))
                    elif category == "sub":
                        subactions.append(parse_action(line))
                    elif category == "conjunction":
                        conjunctions.append(Conjunction(line))
                    elif category == "condition":
                        conditions.append(parse_condition(line))
    return actions, subactions, conjunctions, conditions


def parse_action(line):
    parts = line.split()
    action_dsl_text = parts[0]
    types = []
    for part in parts[1:]:
        if part.isupper():
            action_dsl_text += " {}".format(part)
        else:
            input_match = re.search(r"([\[(])(,?)(\"?)(.+?)(\"?)([])])", part)
            if input_match:
                match_groups = input_match.groups()
                content = match_groups[3]
                if match_groups[0] != match_groups[-1]:
                    raise RuntimeError("Input brackets need to be the same! Given: {}".format(input_match.string))

                if match_groups[2] == '"':
                    if match_groups[2] != match_groups[-2]:
                        raise RuntimeError('The closing quotation mark for this complex input is missing. Given: {}'
                                           .format(input_match.string))
                    type, format = parse_complex_input(content)
                else:
                    type, format = parse_input(content)

                if match_groups[1] == ",":
                    type = (Type.LIST, type)
                    format = "," + format

                types.append(type)

    return Action(action_dsl_text, types)


def parse_input(content):
    format = "{}"
    if content.isupper():
        type = (Type.ACTION, content)
    else:
        type = None
        for input_type in Type:
            if input_type.value == content:
                type = input_type
        if type is None:
            raise RuntimeError("{} is not a valid input type!".format(content))
    return type, format


def parse_complex_input(content):
    format = content
    types = []
    for input_type in Type:
        if input_type.value in format:
            format.replace(input_type.value, "{}")
            types.append(input_type)
    return tuple(types), format


def extract_category(match, in_category):
    category = None
    if match and "/" not in match.group(1):
        if in_category:
            raise RuntimeWarning("The parse script cannot parse subcategories.\n" +
                                 "Please make sure that parts that are surrounded by tags of form '<.+>'" +
                                 "do not contain any further tags")
        in_category = True
        category = match.group(1)
    elif match and "/" in match.group(1):
        if not in_category:
            raise RuntimeError("The parser did not find the start of this category please check that" +
                               " the grammar file has categories enclosed " +
                               "in tags of form '<.+> your_commands </.+>'.")
        in_category = False
        category = None

    return in_category, category
