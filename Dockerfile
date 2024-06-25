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

RUN apt-get install wget  -y
RUN mkdir -p /root/.cache/clip
RUN wget -O /root/.cache/clip/ViT-B-32.pt "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt"

COPY . .