import os
import re
from pathlib import Path

from smart_ccg.combinator.model.dsl.DSL import DSL
from smart_ccg.combinator.model.dsl.dsl_entities.ProtoAction import ProtoAction
from smart_ccg.combinator.model.dsl.dsl_entities.Type import Type


def parse_dsl_file_from(filepath):
    actions = []
    subactions = []
    conjunctions = []
    condition = None
    conditionals = []

    with filepath.open("r") as dsl_file:
        comment = False
        in_category = False
        category = None
        for line in dsl_file:
            if line == '"""\n':
                comment = not comment
            if not comment:
                match = re.search("<(.+?)>", line)
                if match:
                    in_category = not in_category
                    if in_category and category is None:
                        category = match.group(1)
                    elif not in_category and category is not None:
                        category = None
                    else:
                        raise RuntimeError("Nested categories are not supported!")
                elif not match and in_category:
                    line = line.strip()
                    line = line.strip("\t")
                    if category == "complete":
                        actions.append(parse_action(line))
                    elif category == "sub":
                        subactions.append(parse_action(line))
                    elif category == "conjunction":
                        conjunctions.append(line)
                    elif category == "condition":
                        condition = line
                    elif category == "conditionals":
                        conditionals.append(line)

    return DSL(actions, subactions, conjunctions, condition, conditionals)


def parse_action(line):
    parts = line.split()
    action_dsl_texts = []
    input_formats = []
    types = []
    for part in parts:
        match = re.search(r"([\[(])(,?)(.+?)([])])", part)
        if part.isupper() and part.isalpha():
            action_dsl_texts.append(part)
        elif match:
            brackets = (match.group(1), match.group(4))
            is_list = match.group(2) == ","
            is_assignment = "=" in match.group(3)
            if is_assignment:
                assignment_parts = match.group(3).split("=")
                member_types = []
                if len(assignment_parts) != 2:
                    raise RuntimeError("An assignment can only have 2 members. Given: {}"
                                       .format(match.group(3)))
                for member in assignment_parts:
                    member_types.append(extract_input_type(member))
                input_type = (Type.ASSIGNMENT, tuple(member_types))
            else:
                input_type = extract_input_type(match.group(3))

            if is_list:
                input_type = (Type.LIST, input_type)

            if brackets[0] != "[" and brackets[1] != "]":
                input_format = "{}".join(brackets)
            else:
                input_format = "{}"

            input_formats.append(input_format)
            types.append(input_type)
    return ProtoAction(" ".join(action_dsl_texts), types, input_formats)


def extract_input_type(input_text):
    if input_text.isupper():
        return Type.ACTION, input_text
    else:
        for entry in Type:
            if input_text == entry.value:
                return entry


if __name__ == '__main__':
    print(os.getcwd())
    path = Path("./resources/combinator/small.dsl")
    dsl = parse_dsl_file_from(path)
    print([(action.dsl_text, action.input_types) for action in dsl.actions])
    print([(action.dsl_text, action.input_types) for action in dsl.subactions])
    print(dsl.conjunctions)
    print(dsl.condition)
    print(dsl.conditionals)
