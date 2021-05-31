import tkinter as tk

from smart_ccg.combinator.model.Combinator import Combinator
from smart_ccg.combinator.model.dsl.parser.dsl_parser import parse_dsl_file_from


class CombinatorView:
    def __init__(self, combinator):
        self.save_button = tk.Button(
            text="Save",
            width=15,
            height=5,
            bg="lightgrey",
            command=self.on_save
        )
        self.new_example_button = tk.Button(
            text="New Example",
            width=15,
            height=5,
            bg="lightgrey",
            command=self.on_new_example
        )
        self.natural_sentence_label = tk.Label(text="Enter a new natural language sentence here")
        self.natural_sentence_entry = tk.Entry()
        self.combinator = None
        self.natural_sentence = None
        self.table = None

    @classmethod
    def on_create(cls, dsl_filepath, tables_filepath):
        dsl = parse_dsl_file_from(dsl_filepath)
        tables = parse_tables_from(tables_filepath)
        combinator = Combinator(dsl, tables)
        return cls(combinator)




