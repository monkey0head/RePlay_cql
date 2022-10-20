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
        zsh \
    ; \
    rm -rf /var/lib/apt/lists/*;

# set zsh as default shell
SHELL ["/bin/zsh", "-c"]

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
RUN sh -c "$(wget -O- https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"; \
    wget -O mambaforge.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"; \
    bash mambaforge.sh -b; \
    rm mambaforge.sh;
ENV PATH="/root/mambaforge/bin":$PATH

WORKDIR /app
# RUN git clone https://github.com/sb-ai-lab/RePlay.git replay
RUN git clone https://github.com/pkuderov/RePlay.git replay

WORKDIR /app/replay

# Prepare conda env
RUN mamba init zsh; \
    source /root/.zshrc; \
    mamba create --name recsys python=3.9 pip wheel poetry cython -y; \
    mamba activate recsys; \
    # auto activate env on login
    echo "mamba activate recsys" >> /root/.zshrc; \
#     mamba install pytorch -c pytorch -y; \
    mamba install requests pypandoc optuna tabulate -y; \
    pip install datatable; \
    poetry install; \
    pip install -U d3rlpy rs_datasets pytorch_ranger; \
    mamba clean -a -y
