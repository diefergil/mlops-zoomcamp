FROM python:3.9.7-slim

RUN pip install -U pip
RUN pip install pipenv 

WORKDIR /app

COPY [ "Pipfile", "Pipfile.lock", "./"]
RUN pipenv install --system --deploy

COPY ["model.bin",  "predict.py", "./"]

EXPOSE 9696

ENTRYPOINT ["python","./predict.py"]