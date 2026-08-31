"""Thin DuckDB wrapper: the pipeline's analytical store.

One file on disk, no server. Raw pulls land here as tables; downstream steps
read them back with SQL or as DataFrames.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _conn(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(str(self.path))

    def write(self, table: str, df: pd.DataFrame, *, replace: bool = True) -> int:
        with self._conn() as con:
            con.register("_df", df)
            verb = "CREATE OR REPLACE TABLE" if replace else "INSERT INTO"
            if replace:
                con.execute(f"{verb} {table} AS SELECT * FROM _df")
            else:
                con.execute(f"INSERT INTO {table} SELECT * FROM _df")
            n = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        return int(n)

    def read(self, table: str) -> pd.DataFrame:
        with self._conn() as con:
            return con.execute(f"SELECT * FROM {table}").df()

    def sql(self, query: str) -> pd.DataFrame:
        with self._conn() as con:
            return con.execute(query).df()

    def tables(self) -> list[str]:
        with self._conn() as con:
            rows = con.execute("SHOW TABLES").fetchall()
        return [r[0] for r in rows]
