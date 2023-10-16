FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04

# USER root:root

RUN apt-get update && \
    apt-get install -y \
    ffmpeg && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

COPY conda.yaml /tmp/conda.yaml
RUN conda env create -n myenv -f /tmp/conda.yaml
RUN conda activate myenv

# USER dockeruser
