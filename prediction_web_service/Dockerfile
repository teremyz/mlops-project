FROM python:3.7.6-slim

RUN apt-get update && apt-get install -y gcc

RUN pip install -U pip
RUN pip install pipenv

RUN mkdir artifacts

WORKDIR /app

COPY ["predict_api.py", "utils.py", "Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system

ENTRYPOINT ["uvicorn", "predict_api:app", "--host", "0.0.0.0", "--port", "80"]

EXPOSE 80
