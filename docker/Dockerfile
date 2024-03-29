FROM tensorflow/tensorflow:2.11.0-gpu-jupyter

MAINTAINER shaswot <shaswot@acsl.ics.keio.ac.jp>

# install system-wide deps for python and node
# The yqq flag is used to suppress output and assumes "Yes" to all prompts.

# https://github.com/NVIDIA/nvidia-docker/issues/1632
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub
RUN apt-get -yqq update --fix-missing

RUN apt-get -yqq install \
        screen \
        vim \
        git \
        openssh-server \
        htop \
        wget \
        net-tools \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/*

# set a directory for the repos 
RUN mkdir -p /repos/mixedpp

# copy files to the container
COPY . /repos/mixedpp
COPY ./docker/requirements.txt /tmp/

# install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# folder to share files with host machine
RUN mkdir -p /stash
VOLUME /stash


# define the port number the container should expose
# All running containers expose 8888
# They are redirected to other ports using docker -p <machine_port>:8888
EXPOSE 8888

# Change workdir to root so that Jupyter Lab can access the /stash folder
# /stash folder stores large dumpfiles that can be accessed from the host computer
WORKDIR /


# create non-root user
RUN groupadd -g 999 appuser && \
    useradd -r -u 999 -g appuser appuser
USER appuser

# run bash shell
# start jupyter lab once inside the container using screen
# this allows you to access the bash shell and run multiple screens for multiple notebooks / tensorboards 
RUN /bin/bash
