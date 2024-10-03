FROM python:3.11

RUN pip install fastapi[standard]

RUN pip install torchtext==0.17.0

RUN pip install nltk

COPY . /app

WORKDIR /app



ENTRYPOINT [ "fastapi", "run", "api.py"]