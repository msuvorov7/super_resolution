# super_resolution

Application for MLOps course from ODS

## Snakemake run
- `snakemake -n` 
- `snakemake -p`

## DVC
- `dvc repro` (run pipeline)
- `dvc dag` (plot dag in console)

## MLflow
- `mlflow ui` (start mlflow from local dir)
- Tracking server at localhost:
  - `brew install postgres`
  - `pg_ctl -D /usr/local/var/postgres start`
  - `psql postgres`
    - `CREATE DATABASE mlflow_db;`
    - `CREATE USER mlflow_user WITH ENCRYPTED PASSWORD 'mlflow';`
    - `GRANT ALL PRIVILEGES ON DATABASE mlflow_db TO mlflow_user;`
    - `GRANT ALL PRIVILEGES ON DATABASE mlflow_db TO <username>;`
  - `pip install psycopg2`
  - `pg_ctl -D /usr/local/var/postgres restart`
  - `mlflow server --backend-store-uri postgresql://<username>:mlflow@localhost/mlflow_db --default-artifact-root ./mlruns/artifacts/ --host 0.0.0.0 -p 8000`
