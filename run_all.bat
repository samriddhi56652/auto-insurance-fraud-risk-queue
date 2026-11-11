@echo off
REM Full pipeline, Windows cmd. Run from the repo root with the venv activated.
setlocal
for /f "usebackq tokens=1,2 delims==" %%a in (".env") do set %%a=%%b
if "%GCP_PROJECT%"=="" (echo Set GCP_PROJECT in .env & exit /b 1)

echo == BigQuery datasets (no-op if they exist)
for %%d in (fraud_raw fraud_clean fraud_marts) do call bq --location=%BQ_LOCATION% mk -f --dataset %GCP_PROJECT%:%%d

echo == Part B: generate synthetic panel
python -m src.generate_panel || exit /b 1

echo == Load raw (unchanged, all STRING)
python -m src.load_raw a || exit /b 1
python -m src.load_raw b || exit /b 1

echo == SQL: profile -^> clean -^> features -^> checks
python -m src.run_sql a || exit /b 1
python -m src.run_sql b || exit /b 1

echo == Models, panel econometrics, exports
python -m src.part_a_model || exit /b 1
python -m src.part_b_model || exit /b 1
python -m src.part_b_panel || exit /b 1
python -m src.export || exit /b 1

echo == Checkpoint tests
python -m pytest -q
