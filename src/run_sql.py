"""Run numbered SQL files in order. SELECT outputs are saved under
reports/profiling/<part>/ so every number quoted later has a source file.

Run:  python -m src.run_sql b                 (all files in sql/part_b)
      python -m src.run_sql b 01_profile_raw  (just one file)
"""
import sys

from src import bq, config


def main(part: str, only: str | None = None):
    folder = config.ROOT / "sql" / f"part_{part}"
    for path in sorted(folder.glob("*.sql")):
        if only and path.stem != only:
            continue
        print(f"-> {path.relative_to(config.ROOT)}")
        bq.run_file(path, save_results_to=config.PROFILING / f"part_{part}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
