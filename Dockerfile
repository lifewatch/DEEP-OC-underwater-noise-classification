# Dockerfile may have two Arguments: tag, branch
# tag - tag for the Base image, (e.g. 1.10.0-py3 for tensorflow)
# pyVer - python versions as 'python' or 'python3' (default: python3)
# branch - user repository branch to clone (default: master, other option: test)

# Use the official PyTorch image with version 2.1.0 and Python 3
ARG tag=2.1.0-cuda11.8-cudnn8-runtime
FROM pytorch/pytorch:${tag}

# Metadata
LABEL maintainer="Wout Decrop & Ignacio Heredia (CSIC)"
LABEL version="0.1"

# Set environment variables (optional)
ENV DEBIAN_FRONTEND=noninteractive
# An audio classifier with Deep Neural Networks

# What user branch to clone (!)
ARG branch=master
# If to install JupyterLab
ARG jlab=true
# Oneclient version


RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        curl \
        wget \
        psmisc \
        python3-setuptools \
        python3-pip \
        python3-wheel && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /root/.cache/pip/* && \
    rm -rf /tmp/* && \
    python --version && \
    pip --version


# Set LANG environment
ENV LANG C.UTF-8

# Set the working directory
WORKDIR /srv

# Install rclone
RUN wget https://downloads.rclone.org/rclone-current-linux-amd64.deb && \
    dpkg -i rclone-current-linux-amd64.deb && \
    apt install -f && \
    mkdir /srv/.rclone/ && touch /srv/.rclone/rclone.conf && \
    rm rclone-current-linux-amd64.deb && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /root/.cache/pip/* && \
    rm -rf /tmp/*

# * allows to run shorter command "deep-start"
# * allows to install jupyterlab or code-server (vscode),
#   if requested during container creation
RUN git clone https://github.com/deephdc/deep-start /srv/.deep-start && \
    ln -s /srv/.deep-start/deep-start.sh /usr/local/bin/deep-start

# # Install DEEPaaS from PyPi
# RUN pip install --no-cache-dir deepaas && \
#     rm -rf /root/.cache/pip/* && \
#     rm -rf /tmp/*

RUN pip install --upgrade setuptools
RUN pip install --upgrade pip

RUN pip install --no-cache-dir flaat cachetools==4.* && \
    rm -rf /root/.cache/pip/* && \
    rm -rf /tmp/*

# Install FLAAT (FLAsk support for handling Access Tokens)
RUN pip install --no-cache-dir flaat && \
    rm -rf /root/.cache/pip/* && \
    rm -rf /tmp/*

# Disable FLAAT authentication by default
ENV DISABLE_AUTHENTICATION_AND_ASSUME_AUTHENTICATED_USER yes

# Install DEEPaaS from PyPi:
RUN pip install --no-cache-dir deepaas && \
    rm -rf /root/.cache/pip/* && \
    rm -rf /tmp/*

# Useful tool to debug extensions loading
RUN pip install --no-cache-dir entry_point_inspector && \
    rm -rf /root/.cache/pip/* && \
    rm -rf /tmp/*


# Install JupyterLab
# ENV JUPYTER_CONFIG_DIR /srv/.jupyter/
ENV SHELL /bin/bash
# Install audio packages
RUN apt update && \
    apt install -y ffmpeg libavcodec-extra

# Install user app
RUN git clone -b $branch https://github.com/lifewatch/DEEP-OC-underwater-noise-classification && \
    cd  underwater-noise-classification && \
    pip install --no-cache-dir -e . && \
    rm -rf /root/.cache/pip/* && \
    rm -rf /tmp/* && \
    cd ..

# Open DEEPaaS port
EXPOSE 5000

# Open Monitoring port
EXPOSE 6006

# Open JupyterLab port
EXPOSE 8888

# Account for OpenWisk functionality (deepaas >=0.4.0) + proper docker stop
CMD ["deepaas-run", "--listen-ip", "0.0.0.0", "--listen-port", "5000"]
