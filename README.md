# super_resolution

Application for MLOps course from ODS

[Description](https://yandex.ru/q/article/proekt_po_kursu_mlops_6a54c39d/) on Yandex/Q

## Snakemake run
- `snakemake -n` 
- `snakemake -p`

## DVC
- `dvc repro` (run pipeline)
- `dvc dag` (plot dag in console)

## MLflow
- `mlflow ui` (start mlflow from local dir)
- Tracking server at localhost:
  - postgres:
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
  - sqlite:
    - `mlflow server --backend-store-uri sqlite:///mydb.sqlite --default-artifact-root ./mlruns/artifacts/ --host localhost`
  
  (don't forget change `mlflow.set_tracking_uri` in `train_model.py`)

## MLflow in docker
- create .env file like .env.example in project_root_dir
- mlflow uses boto3 to get access for s3 storage, you need to:
  - `mkdir ~/.aws`
  - `nano ~/.aws/credentials`
  ```
  [default]
  aws_access_key_id=AS_IN_.ENV
  aws_secret_access_key=AS_IN_.ENV
  aws_bucket_name=AS_IN_.ENV
  ```
  - `MLFLOW_S3_ENDPOINT_URL=http://localhost:9000` very IMPORTANT (.env)

## FastAPI
- `docker build -f Docker/mlflow_image/Dockerfile -t mlflow_server .`
- `docker build -f Docker/model_service/Dockerfile -t model_service .`
- `docker-compose up -d --build`

Address: 127.0.0.1:8000

## Nexus
- `docker exec -it nexus bash`
  - copy pass from /nexus-data/admin.password
- sign in 127.0.0.1:8082 (login:admin)
- create docker(hosted) repository with HTTP port 8123
- `docker login -u admin -p <password> 127.0.0.1:8123`
- `docker tag model_service:latest 127.0.0.1:8123/model_service:v1`
- `docker push 127.0.0.1:8123/model_service:v1`
  