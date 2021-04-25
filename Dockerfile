FROM ubuntu:latest

# setup gcc and other compilers

RUN apt-get update && \
    apt-get -y install gcc mono-mcs make cmake && \
    rm -rf /var/lib/apt/lists/*

# install libraries and other dependencies

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y wget \
    && apt-get install -y zlib1g-dev \
  	&& rm -rf /var/lib/apt/lists/*

RUN apt-get update \
    && apt-get -y install libssl-dev libncurses5-dev libsqlite3-dev libreadline-dev libffi-dev libopenmpi-dev

RUN apt-get -y install openssl

# install python3.8 and pip

RUN apt -y install software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get -y install python3.8
RUN apt-get install -y python3-pip
RUN pip3 install --upgrade pip

# install git

RUN apt-get update && apt-get install -y git

# clone obstacle-tower-challenge repo

RUN git clone https://github.com/cstrojans/obstacle-tower-challenge.git && cd obstacle-tower-challenge && git checkout master

# install our game dependencies

RUN pip3 install -r obstacle-tower-challenge/requirements.txt