FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

RUN apt-get update && apt-get install -y software-properties-common
RUN apt-get update && apt-get install -y && \
    pip install --upgrade pip
RUN apt-get update \
    && apt-get install -yqq \
        git git-core

COPY requirements.txt /tmp/requirements-docker.txt

ENV INPUT_DATA="./data/input"
ENV OUTPUT_DATA="./data/output"
ENV DATASET_ID="nasa_rwanda_field_boundary_competition"

RUN pip install -r /tmp/requirements-docker.txt && \
    rm /tmp/requirements-docker.txt

RUN mkdir -p /app
WORKDIR /app

COPY . .

ENTRYPOINT [ "bash", "run_model.sh" ]