import tkinter as tk

import pandas as pd
from sortedcontainers import SortedList


class DataFrameTable(tk.Frame):
    def __init__(self, parent, indices, column_heads, background="black"):
        tk.Frame.__init__(self, parent, background=background)
        self.columns = {column_head: column for (column_head, column)
                        in zip(column_heads, range(len(column_heads)))}
        self.rows = SortedList(indices)
        self.value_widgets = []

        for index in self.rows:
            current_row = []
            for (column_head, column) in self.columns:
                label = tk.Label(self, text="%s/%s" % (index, column_head), borderwidth=0,
                                 width=10)
                label.grid(row=self.rows.index(index), column=column, sticky="nsew", padx=1, pady=1)
                current_row.append(label)
            self.value_widgets.append(current_row)

        for column in self.columns.values():
            self.grid_columnconfigure(column, weight=1)

    @classmethod
    def create_table_from_dataframe(cls, parent: tk.Tk, data_frame:  pd.DataFrame, background="black"):
        table = DataFrameTable(parent, data_frame.index.tolist(), data_frame.columns.tolist(), background=background)
        for index in data_frame.index:
            table.set_row(index, data_frame.iloc(index).tolist())

    def set_row(self, index, row_list):
        try:
            row = self.rows.index(index)
        except ValueError:
            self.rows.append(index)
            row = self.rows.index(index)

        for column in self.columns.values():
            value = row_list[column]
            self._set(row, column, value)

    def set_cell(self, index, column_label, value):
        row = self.rows.index(index)
        column = self.columns[column_label]
        self._set(row, column, value)

    def _set(self, row, column, value):
        widget = self.value_widgets[row][column]
        widget.configure(text=value)
