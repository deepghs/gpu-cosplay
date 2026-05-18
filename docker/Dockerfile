# syntax=docker/dockerfile:1.4
#
# Two ways to use this Dockerfile:
#
#   1. Default: starts from nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04 and
#      adds sshd, sudo, tini, and the gpu-cosplay entrypoint. ~5 GB image.
#      Built by `gpu-cosplay build` with no arguments.
#
#   2. Layer on top of your own image (recommended when you already have a
#      tuned PyTorch/training environment):
#          gpu-cosplay build --base my-org/pytorch:v3 --tag my-cosplay-pt
#      Your image keeps all its installed wheels/datasets; we just add ~200 MB
#      of sshd + helpers. Then:
#          gpu-cosplay up 3090 --image my-cosplay-pt
#
# The BASE_IMAGE must be Ubuntu/Debian-derived (we use apt-get).
ARG BASE_IMAGE=nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PYTHONPATH=/opt/gpu-cosplay/python:${PYTHONPATH:-}

RUN apt-get update && apt-get install -y --no-install-recommends \
        openssh-server sudo ca-certificates curl wget git tini \
        python3 python3-pip python3-venv \
        build-essential pkg-config \
        vim less tmux htop \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /var/run/sshd /workspace /opt/gpu-cosplay/python \
    && sed -i 's/#PermitRootLogin .*/PermitRootLogin no/' /etc/ssh/sshd_config \
    && sed -i 's/#PasswordAuthentication .*/PasswordAuthentication no/' /etc/ssh/sshd_config \
    && sed -i 's/#PubkeyAuthentication .*/PubkeyAuthentication yes/' /etc/ssh/sshd_config

COPY entrypoint.sh             /entrypoint.sh
COPY gpu_cosplay_inject.py     /opt/gpu-cosplay/python/gpu_cosplay_inject.py
COPY gpu_cosplay_apply.py      /usr/local/bin/gpu-cosplay-apply
RUN chmod +x /entrypoint.sh /usr/local/bin/gpu-cosplay-apply

EXPOSE 22
WORKDIR /workspace
ENTRYPOINT ["/usr/bin/tini","--","/entrypoint.sh"]
CMD ["/usr/sbin/sshd","-D","-e"]
