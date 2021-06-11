import tkinter as tk

from smart_ccg.combinator.model.Combinator import Combinator
from smart_ccg.combinator.util.wikitablequestions_table_parser import parse_wikitablequestions_tables_from
from smart_ccg.combinator.util.dsl_parser import parse_dsl_file_from
from smart_ccg.combinator.ui.DataFrameTable import DataFrameTable


class CombinatorView(tk.Tk):
    def __init__(self, combinator):
        tk.Tk.__init__(self)
        self.configure_menubar()
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
        self.combinator = combinator
        self.dsl_output = tk.Label()
        self.natural_sentence_label = tk.Label(text="Enter a new natural language sentence here")
        self.natural_sentence_entry = tk.Entry()
        self.natural_sentence = None
        self.table = tk.Frame()


    @classmethod
    def configure_menubar(cls):
        cls.menubar = tk.Menu(cls)
        cls.options_menu = tk.Menu(cls.menubar, tearoff=0)
        cls.options_menu.add_command(label="Settings", command=cls.on_open_settings)
        cls.options_menu.add_separator()
        cls.options_menu.add_command(label="Exit", command=cls.quit)
        cls.menubar.add_cascade(label="Options", menu=cls.options_menu)

    @classmethod
    def create_instance(cls, storage, dsl_filepath, wikitablequestions_path):
        dsl = parse_dsl_file_from(dsl_filepath)
        tables = parse_wikitablequestions_tables_from(wikitablequestions_path)
        combinator = Combinator(dsl, tables)
        combinator_view = cls(combinator)
        combinator_view.pack()
        return combinator_view

    def pack(self):
        self.natural_sentence_label.pack()
        self.natural_sentence_entry.pack()
        self.save_button.pack()
        self.new_example_button.pack()
        self.table.pack()
        self.menubar.pack()

    def on_save(self):
        pass

    def on_new_example(self):
        dsl_output, table = self.combinator.generate_randomized_example()
        self.dsl_output.configure(text=dsl_output)
        self.table = DataFrameTable.create_table_from_dataframe(self, table)
        self.table.pack()

    def on_open_settings(self):
        pass
