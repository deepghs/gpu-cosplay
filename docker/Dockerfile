# syntax=docker/dockerfile:1.4
#
# Two ways to use this Dockerfile:
#
#   1. Default: nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04 + sshd + sudo + tini
#      + the gpu-cosplay runtime hooks. ~5 GB image. Built by `gpu-cosplay build`.
#
#   2. Layered on your own image:
#          gpu-cosplay build --base my-org/pytorch:v3 --tag my-cosplay-pt
#      Your image keeps its installed wheels/datasets; we add ~200 MB of sshd +
#      helpers. Then: `gpu-cosplay up 3090 --image my-cosplay-pt`.
#
# BASE_IMAGE must be Ubuntu/Debian-derived (we use apt-get).
ARG BASE_IMAGE=nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1

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

# Runtime hooks (.pth + module). The .pth gets symlinked into the live python's
# site-packages by entrypoint.sh so it works for system python, conda, venvs.
COPY entrypoint.sh                   /entrypoint.sh
COPY gpu_cosplay_runtime.py          /opt/gpu-cosplay/python/gpu_cosplay_runtime.py
COPY gpu_cosplay_runtime.pth         /opt/gpu-cosplay/python/gpu_cosplay_runtime.pth
# nvidia-smi shim — lives at /usr/local/bin/nvidia-smi which precedes /usr/bin.
# The real /usr/bin/nvidia-smi is NEVER touched; the shim calls it via shutil.which.
COPY nvidia-smi                      /usr/local/bin/nvidia-smi
# Self-check tool.
COPY gpu_cosplay_verify.py           /usr/local/bin/gpu-cosplay-verify

RUN chmod +x /entrypoint.sh /usr/local/bin/nvidia-smi /usr/local/bin/gpu-cosplay-verify

EXPOSE 22
WORKDIR /workspace
ENTRYPOINT ["/usr/bin/tini","--","/entrypoint.sh"]
CMD ["/usr/sbin/sshd","-D","-e"]
