# syntax=docker/dockerfile:1.4
ARG CUDA_TAG=12.6.3-cudnn-devel-ubuntu24.04
FROM nvidia/cuda:${CUDA_TAG}

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
    && mkdir -p /var/run/sshd /workspace \
    && sed -i 's/#PermitRootLogin .*/PermitRootLogin no/' /etc/ssh/sshd_config \
    && sed -i 's/#PasswordAuthentication .*/PasswordAuthentication no/' /etc/ssh/sshd_config \
    && sed -i 's/#PubkeyAuthentication .*/PubkeyAuthentication yes/' /etc/ssh/sshd_config

COPY entrypoint.sh /entrypoint.sh
COPY gpu_cosplay_inject.py /usr/local/lib/python3.12/dist-packages/gpu_cosplay_inject.py
COPY gpu_cosplay_apply.py  /usr/local/bin/gpu-cosplay-apply
RUN chmod +x /entrypoint.sh /usr/local/bin/gpu-cosplay-apply

EXPOSE 22
WORKDIR /workspace
ENTRYPOINT ["/usr/bin/tini","--","/entrypoint.sh"]
CMD ["/usr/sbin/sshd","-D","-e"]
