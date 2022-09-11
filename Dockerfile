FROM python:3.9-slim

RUN apt-get update; \
    apt-get install -y \
        ca-certificates \
        wget \
        tar \
        make \
        cmake \
        g++ \
        git \
        xz-utils \
        git \
    ; \
    rm -rf /var/lib/apt/lists/*;


WORKDIR /app
# RUN git clone https://github.com/sb-ai-lab/RePlay.git replay
RUN git clone https://github.com/pkuderov/RePlay.git replay

WORKDIR /app/replay
RUN pip install -U pip wheel; \
    pip install -U requests pypandoc cython optuna poetry; \
    pip install -U d3rlpy rs_datasets; \
    poetry install; \
