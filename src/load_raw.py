"""Load raw CSVs into fraud_raw unchanged (every column STRING) and log
row counts + file hashes to reports/load_log.csv.

Run:  python -m src.load_raw a     (Part A, real Kaggle file)
      python -m src.load_raw b     (Part B, synthetic panel)

The Part B ground-truth file is deliberately NOT loaded to BigQuery: it is
only read by the evaluation code, so it can never leak into the feature SQL.
"""
import csv
import hashlib
import sys
from datetime import datetime, timezone

import pandas as pd

from src import bq, config

SOURCES = {
    "a": {"a_claims": config.PART_A_FILE},
    "b": {f"b_{t}": config.DATA_SYN / f"{t}.csv" for t in ("policyholders", "policy_months", "claims")},
}


def main(part: str):
    log_rows = []
    for table, path in SOURCES[part].items():
        # utf-8-sig strips the byte-order mark the Kaggle file starts with
        with open(path, newline="", encoding="utf-8-sig") as f:
            header = next(csv.reader(f))
            file_rows = sum(1 for _ in f)
        n = bq.load_csv_as_strings(path, bq.table_id(config.DS_RAW, table), header)
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        status = "OK" if n == file_rows else "MISMATCH"
        print(f"{table:18s} file={file_rows:>8,d} bq={n:>8,d} cols={len(header)} {status}")
        log_rows.append(dict(loaded_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                             table=f"{config.DS_RAW}.{table}", source_file=path.name, sha256=sha,
                             file_data_rows=file_rows, bq_rows=n, columns=len(header), check=status))
        if status != "OK":
            raise SystemExit(f"Row count mismatch for {table}")
    log = config.REPORTS / "load_log.csv"
    old = pd.read_csv(log) if log.exists() else pd.DataFrame()
    old = old[~old.get("table", pd.Series(dtype=str)).isin([r["table"] for r in log_rows])] if len(old) else old
    pd.concat([old, pd.DataFrame(log_rows)]).to_csv(log, index=False)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "b")
