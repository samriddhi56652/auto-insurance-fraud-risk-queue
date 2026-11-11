"""Thin BigQuery helpers: run SQL files, load CSVs unchanged, read tables."""
import re
from pathlib import Path

import pandas as pd
from google.cloud import bigquery

from src import config


def client() -> bigquery.Client:
    return bigquery.Client(project=config.GCP_PROJECT, location=config.BQ_LOCATION)


def _job_config(**kw) -> bigquery.QueryJobConfig:
    return bigquery.QueryJobConfig(maximum_bytes_billed=config.BQ_MAX_BYTES_BILLED, **kw)


def render(sql: str) -> str:
    return sql.replace("{project}", config.GCP_PROJECT)


def query(sql: str) -> pd.DataFrame:
    return client().query(render(sql), job_config=_job_config()).to_dataframe()


def split_statements(sql: str) -> list[str]:
    """Split a file on ';' at line ends, ignoring '--' comment-only chunks."""
    parts = re.split(r";\s*\n", sql)
    out = []
    for p in parts:
        body = "\n".join(l for l in p.splitlines() if not l.strip().startswith("--")).strip()
        if body:
            out.append(p.strip().rstrip(";"))
    return out


def run_file(path: Path, save_results_to: Path | None = None) -> list[pd.DataFrame]:
    """Run every statement in a SQL file. SELECT results are returned and, if
    save_results_to is given, written as CSV (one per statement) so every
    profiling number in the README is traceable to a file."""
    results = []
    c = client()
    for i, stmt in enumerate(split_statements(path.read_text(encoding="utf-8")), 1):
        job = c.query(render(stmt), job_config=_job_config())
        rows = job.result()
        if job.statement_type == "SELECT":
            df = rows.to_dataframe()
            results.append(df)
            if save_results_to is not None:
                save_results_to.mkdir(parents=True, exist_ok=True)
                df.to_csv(save_results_to / f"{path.stem}_{i:02d}.csv", index=False)
        print(f"  {path.name} [{i}] {job.statement_type} ok "
              f"({(job.total_bytes_processed or 0) / 1e6:.2f} MB processed)")
    return results


def load_csv_as_strings(csv_path: Path, table: str, columns: list[str]) -> int:
    """Load a CSV into BigQuery with every column as STRING (no type guessing),
    so cleaning rules are explicit in SQL. Returns the loaded row count."""
    schema = [bigquery.SchemaField(c, "STRING") for c in columns]
    job_config = bigquery.LoadJobConfig(
        schema=schema, skip_leading_rows=1, source_format=bigquery.SourceFormat.CSV,
        write_disposition="WRITE_TRUNCATE", allow_quoted_newlines=False,
    )
    c = client()
    with open(csv_path, "rb") as f:
        c.load_table_from_file(f, table, job_config=job_config).result()
    return c.get_table(table).num_rows


def load_dataframe(df: pd.DataFrame, table: str) -> int:
    """Load via CSV, not the client's default parquet path: Windows Smart App Control
    blocks pyarrow's parquet DLL on this machine (see ERRORS.md)."""
    import io
    c = client()
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE", source_format=bigquery.SourceFormat.CSV,
        skip_leading_rows=1, autodetect=True)
    buf = io.BytesIO(df.to_csv(index=False).encode("utf-8"))
    c.load_table_from_file(buf, table, job_config=job_config).result()
    return c.get_table(table).num_rows


def table_id(dataset: str, name: str) -> str:
    return f"{config.GCP_PROJECT}.{dataset}.{name}"
