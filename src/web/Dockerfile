FROM python:3.11

RUN pip install streamlit

COPY . /app
WORKDIR /app
ENTRYPOINT [ "streamlit", "run", "main.py"]