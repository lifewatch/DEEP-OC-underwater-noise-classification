# Dockerfile may have two Arguments: tag, branch
# tag - tag for the Base image, (e.g. 1.10.0-py3 for tensorflow)
# pyVer - python versions as 'python' or 'python3' (default: python3)
# branch - user repository branch to clone (default: master, other option: test)

ARG tag=1.14.0-py3

# Base image, e.g. tensorflow/tensorflow:1.12.0-py3
FROM tensorflow/tensorflow:${tag}

LABEL maintainer='Wout Decrop & Ignacio Heredia (CSIC)'
LABEL version='0.1'
# An audio classifier with Deep Neural Networks

# What user branch to clone (!)
ARG branch=master
# If to install JupyterLab
ARG jlab=true
# Oneclient version
# ARG oneclient_ver=19.02.0.rc2-1~bionic

# RUN apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub

ENV DEBIAN_FRONTEND=noninteractive

# Install ubuntu updates and python related stuff
# link python3 to python, pip3 to pip, if needed
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
         git \
         curl \
         wget \
         psmisc \
         python3-setuptools \
         python3-pip \
         python3-wheel \
         libgl1 \
         libsm6 \
         libxrender1 \
         libfontconfig1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /root/.cache/pip/* \
    && rm -rf /tmp/*

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

# # Install oneclient for ONEDATA
# RUN curl -sS  http://get.onedata.org/oneclient-1902.sh | bash -s -- oneclient="$oneclient_ver" && \
#     apt-get clean && \
#     mkdir -p /mnt/onedata && \
#     rm -rf /var/lib/apt/lists/* && \
#     rm -rf /tmp/*

# Install deep-start script
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





# RUN pip install --no-cache-dir flaat && \
#     rm -rf /root/.cache/pip/* && \
#     rm -rf /tmp/*

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

# Install DEEP debug_log scripts:
# RUN git clone https://github.com/deephdc/deep-debug_log /srv/.debug_log

# Install JupyterLab
# ENV JUPYTER_CONFIG_DIR /srv/.jupyter/
ENV SHELL /bin/bash
# RUN if [ "$jlab" = true ]; then \
#        pip install --no-cache-dir jupyterlab ; \
#        git clone https://github.com/deephdc/deep-jupyter /srv/.jupyter ; \
#     else echo "[INFO] Skip JupyterLab installation!"; fi

# Install audio packages
RUN apt update && \
    apt install -y ffmpeg libavcodec-extra

# Install user app
RUN git clone -b $branch https://github.com/lifewatch/underwater-noise-classification && \
    cd  underwater-noise-classification && \
    pip install --no-cache-dir -e . && \
    rm -rf /root/.cache/pip/* && \
    rm -rf /tmp/* && \
    cd ..

# RUN cd  underwater-noise-classification && \
#     pip install --no-cache-dir -e . && \
#     cd ..

# Download network weights: compressing with tar.xz gives decompression errors (corrupt data)
# ENV SWIFT_CONTAINER https://cephrgw01.ifca.es:8080/swift/v1/audio-classification-tf/
ENV SWIFT_CONTAINER https://api.cloud.ifca.es:8080/swift/v1/audio-classification-tf/
ENV MODEL_TAR default.tar.gz

RUN curl --insecure -o ./underwater-noise-classification/models/${MODEL_TAR} \
    ${SWIFT_CONTAINER}${MODEL_TAR}

RUN cd underwater-noise-classification/models && \
    tar -zxvf ${MODEL_TAR}  && \
    rm ${MODEL_TAR}

# Open DEEPaaS port
EXPOSE 5000

# Open Monitoring port
EXPOSE 6006

# Open JupyterLab port
EXPOSE 8888

# Account for OpenWisk functionality (deepaas >=0.4.0) + proper docker stop
CMD ["deepaas-run", "--listen-ip", "0.0.0.0", "--listen-port", "5000"]
