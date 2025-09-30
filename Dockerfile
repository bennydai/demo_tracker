FROM python:3.11.13

# Install prerequisites
RUN apt-get update && apt-get install -y build-essential

RUN apt-get update && apt-get install -y vim wget

# Create temp directory
WORKDIR /opt/

# Install pip
RUN pip install --upgrade pip setuptools wheel

# Install ffmpeg libraries for opencv
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 \
    libjpeg-dev libpng-dev

# Install requirements for h5
RUN apt-get update && apt-get install -y libhdf5-serial-dev hdf5-tools \
    libhdf5-dev zlib1g-dev zip liblapack-dev libblas-dev gfortran

# Install necessary c++ libraries
RUN apt-get update && apt-get install -y libeigen3-dev
RUN apt-get update && apt-get install -y libopencv-dev
RUN apt-get update && apt-get install -y libboost-all-dev
RUN apt-get update && apt-get install -y libyaml-cpp-dev

# Install requirements.txt
COPY requirements.txt .
RUN pip install -r requirements.txt

# Install gdb debugger
RUN apt-get update && apt-get install -y gdb

# Install additional software
RUN apt-get update && apt-get install -y openssh-client

# Reset
WORKDIR /app/
SHELL ["/bin/bash", "-c"]