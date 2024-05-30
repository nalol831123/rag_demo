FROM nvidia/cuda:11.4.3-cudnn8-devel-ubuntu20.04

WORKDIR /rag_demo

ADD . .

ENV DEBIAN_FRONTEND=noninteractive

# Install Python and pip
RUN apt-get clean
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libgl1-mesa-glx -y \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

#ADD requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir
#RUN pip install -r requirements.txt

RUN pip install unstructured[pdf]

EXPOSE 8501