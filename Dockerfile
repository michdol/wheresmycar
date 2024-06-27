FROM python:3.12-slim-bullseye AS build

ARG PROJECT_NAME=wheresmycar

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apt-get update && \
    apt-get install --no-install-recommends -y \
        libpython3-dev \
        ffmpeg libsm6 libxext6
        gcc && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

CMD mkdir ${PROJECT_NAME}
COPY . ${PROJECT_NAME}
WORKDIR ${PROJECT_NAME}

RUN pip install --no-cache-dir --upgrade -r ${PROJECT_NAME}/requirements.txt

ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1
CMD tail -f /dev/null