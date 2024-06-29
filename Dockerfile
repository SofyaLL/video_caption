FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y curl git
# Installing Rust for tokenizers
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- --default-toolchain=1.72.1 -y
ENV PATH="/root/.cargo/bin:${PATH}"
ENV RUSTUP_TOOLCHAIN=1.72.1

RUN pip3 install --upgrade pip
RUN pip3 install --upgrade setuptools wheel
RUN apt-get install build-essential  -y
RUN apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 

COPY requirements.txt .

RUN pip3 install --no-cache-dir --upgrade -r requirements.txt

RUN python -m pip install git+https://github.com/openai/CLIP.git

COPY . .