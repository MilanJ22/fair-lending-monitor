import duckdb
import pandas as pd

DB_PATH = "data/hmda.duckdb"


def get_connection() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(DB_PATH)


def save_all(
    applications: pd.DataFrame,
    denial_rates: pd.DataFrame,
    peer_benchmarks: pd.DataFrame,
    disparity_flags: pd.DataFrame,
    denial_reasons: pd.DataFrame,
    institutions: pd.DataFrame,
) -> None:
    """
    Write all processed DataFrames to DuckDB. Replaces existing tables on each run.
    """
    con = get_connection()

    tables = {
        "applications": applications,
        "denial_rates": denial_rates,
        "peer_benchmarks": peer_benchmarks,
        "disparity_flags": disparity_flags,
        "denial_reasons": denial_reasons,
        "institutions": institutions,
    }

    for name, df in tables.items():
        con.execute(f"DROP TABLE IF EXISTS {name}")
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM df")
        print(f"  Saved '{name}': {len(df):,} rows")

    con.close()
    print(f"\nDuckDB written to {DB_PATH}")


def query(sql: str) -> pd.DataFrame:
    """Run an arbitrary SQL query against the DuckDB database."""
    con = get_connection()
    result = con.execute(sql).df()
    con.close()
    return result
