FROM ubuntu:18.04

RUN apt-get update
RUN apt-get -y install unzip git wget python3-pip patchelf libopenmpi-dev libosmesa-dev libgl1-mesa-dev swig libglfw3-dev python-opengl ffmpeg tmux xvfb
RUN pip3 install pipenv

RUN mkdir -p /root/.mujoco \
    && wget https://www.roboti.us/download/mjpro150_linux.zip -O mujoco150.zip \
    && unzip mujoco150.zip -d /root/.mujoco \
    && rm mujoco150.zip
ENV LD_LIBRARY_PATH /root/.mujoco/mjpro150/bin

ARG MUJOCO_KEY
RUN wget -O /root/.mujoco/mjkey.txt ${MUJOCO_KEY}

COPY docker_entrypoint /entrypoint
ENTRYPOINT ["/entrypoint"]

# needed for pipenv
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

COPY . /interactive-behaviour-design

# This isn't a perfect solution: different branches could have their own versions of the submodules
# which could have different requirements. But this is good enough for now.
RUN cd interactive-behaviour-design && pipenv run pip install tensorflow==1.13.1 && pipenv sync
# Fix "AttributeError: module '_Box2D' has no attribute 'RAND_LIMIT_swigconstant'"
RUN cd interactive-behaviour-design && pipenv run pip uninstall -y box2d-py && pipenv run pip install box2d-py
RUN cd interactive-behaviour-design/baselines && pipenv run pip install -e .
RUN cd interactive-behaviour-design/spinningup && pipenv run pip install -e .
# spinup installs a version of gym which is too new, so we uninstall it first
RUN cd interactive-behaviour-design && pipenv run pip uninstall -y gym && cd gym && pipenv run pip install -e .

# We download code from S3; remove current code to avoid any confusion
# (Keep Pipfile so we can still activate the virtualenv)
RUN find interactive-behaviour-design -maxdepth 1 -mindepth 1 -not -name 'Pipfile' -exec rm -rf {} \;
