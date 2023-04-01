FROM python:3.7.6-slim

RUN apt-get update && apt-get install -y gcc

RUN pip install -U pip
RUN pip install pipenv

WORKDIR /app

RUN mkdir data/

COPY [ "data/Train.csv", "data/Test.csv", "data/" ]

COPY [ "Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --system

RUN pip install s3fs
RUN pip install gcsfs
RUN pip install adlfs

COPY [ "train_model.py", "deploy_train_model.py", "deploy_and_agent.sh", ".prefectignore", "./" ]

ENTRYPOINT ["./deploy_and_agent.sh"]
