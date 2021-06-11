from pathlib import Path

import pandas as pd


def parse_wikitablequestions_tables_from(path: Path):
    tables = []
    for subpath in path.rglob("*"):
        if subpath.is_file() and subpath.suffix == ".csv":
            tables.append(pd.read_csv())
    return tables
