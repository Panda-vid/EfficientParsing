from pathlib import Path
from typing import List

import pandas as pd


def parse_wikitablequestions_tables_from(path: Path) -> List[pd.DataFrame]:
    tables = []
    for subpath in path.rglob("*"):
        tables.append(parse_csv_table(subpath))
    return tables


def parse_csv_table(path: Path) -> pd.DataFrame:
    if path.is_file() and path.suffix == ".csv":
        return pd.read_csv(path)
