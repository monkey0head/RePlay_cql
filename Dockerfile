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
        # for: javasdk
        apt-transport-https \
    ; \
    # regular clean-up
    rm -rf /var/lib/apt/lists/*;

# Install Java 11 (or 8) SDK
RUN mkdir -p /etc/apt/keyrings; \
    wget -O - https://packages.adoptium.net/artifactory/api/gpg/key/public \
        | tee /etc/apt/keyrings/adoptium.asc; \
    echo "deb [signed-by=/etc/apt/keyrings/adoptium.asc] https://packages.adoptium.net/artifactory/deb $(awk -F= '/^VERSION_CODENAME/{print$2}' /etc/os-release) main" \
        | tee /etc/apt/sources.list.d/adoptium.list; \
    apt-get update; \
    apt-get install -y temurin-11-jdk; \
    rm -rf /var/lib/apt/lists/*;

# Install oh-my-zsh and mambaforge as environment management
RUN wget -O mambaforge.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"; \
    bash mambaforge.sh -b; \
    rm mambaforge.sh;
ENV PATH="/root/mambaforge/bin":$PATH

WORKDIR /
# RUN git clone https://github.com/sb-ai-lab/RePlay.git replay
RUN git clone https://github.com/pkuderov/RePlay.git replay

WORKDIR /replay

# set bash as default shell
SHELL ["/bin/bash", "-c"]

# Prepare conda env
RUN \
    # mandatory to do init + sourcing as each RUN is a new shell session
    mamba init; \
    source /root/.bashrc; \
    # not necessary, just make using base env explicit
    mamba activate base; \
    mamba install python=3.9 pip wheel poetry=1.1 cython requests pypandoc optuna tabulate -y; \
    pip install --no-cache-dir datatable; \
    # prevent poetry to create a separate virtualenv
    poetry config virtualenvs.create false; \
    poetry install; \
    # install non-replay dev dependencies
    mamba install ruamel.yaml -y; \
    pip install --no-cache-dir -U d3rlpy rs_datasets pytorch_ranger wandb; \
    # clean-up
    mamba clean -a -y; \
    pip cache purge;
