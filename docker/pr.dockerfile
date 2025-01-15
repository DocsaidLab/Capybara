# syntax=docker/dockerfile:experimental
FROM nvcr.io/nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Taipei

RUN apt-get update -y && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    tzdata wget git libturbojpeg exiftool ffmpeg poppler-utils libpng-dev \
    libtiff5-dev libjpeg8-dev libopenjp2-7-dev zlib1g-dev gcc \
    libfreetype6-dev liblcms2-dev libwebp-dev tcl8.6-dev tk8.6-dev python3-tk \
    python3-pip libharfbuzz-dev libfribidi-dev libxcb1-dev libfftw3-dev gosu \
    libpq-dev python3-dev && \
    ln -sf /usr/share/zoneinfo/$TZ /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
