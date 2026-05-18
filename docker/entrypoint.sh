#!/bin/bash
# gpu-cosplay container entrypoint.
#
# Responsibilities:
#  1. Create a non-root user matching the host's $HOST_USER/$HOST_UID/$HOST_GID
#     so files in /workspace stay owned consistently across the boundary.
#  2. Install the host's SSH pubkey into authorized_keys.
#  3. Generate sshd host keys if missing.
#  4. Run the requested command (default: sshd -D).
set -euo pipefail

HOST_USER="${HOST_USER:-ubuntu}"
HOST_UID="${HOST_UID:-1000}"
HOST_GID="${HOST_GID:-1000}"

# Group: create if missing; if the gid is taken by another group, reuse that name
if ! getent group "${HOST_GID}" >/dev/null; then
    groupadd -g "${HOST_GID}" "${HOST_USER}" 2>/dev/null \
        || groupadd -g "${HOST_GID}" "cosplay-${HOST_GID}"
fi
GROUP_NAME="$(getent group "${HOST_GID}" | cut -d: -f1)"

# User: create if missing
if ! id -u "${HOST_USER}" >/dev/null 2>&1; then
    if getent passwd "${HOST_UID}" >/dev/null; then
        # Rename the existing user to HOST_USER so uid matches
        existing="$(getent passwd "${HOST_UID}" | cut -d: -f1)"
        usermod -l "${HOST_USER}" -d "/home/${HOST_USER}" -m "${existing}" 2>/dev/null || true
        groupmod -n "${HOST_USER}" "$(id -gn "${HOST_USER}")" 2>/dev/null || true
    else
        useradd -m -u "${HOST_UID}" -g "${HOST_GID}" -s /bin/bash "${HOST_USER}"
    fi
fi

# Sudo without password (convenience inside the cosplay container)
echo "${HOST_USER} ALL=(ALL) NOPASSWD: ALL" >/etc/sudoers.d/90-cosplay
chmod 440 /etc/sudoers.d/90-cosplay

# SSH dir + authorized_keys
SSH_DIR="/home/${HOST_USER}/.ssh"
mkdir -p "${SSH_DIR}"
chmod 700 "${SSH_DIR}"
if [[ -n "${GPU_COSPLAY_PUBKEY:-}" ]]; then
    echo "${GPU_COSPLAY_PUBKEY}" > "${SSH_DIR}/authorized_keys"
    chmod 600 "${SSH_DIR}/authorized_keys"
fi
chown -R "${HOST_USER}:${GROUP_NAME}" "${SSH_DIR}" "/home/${HOST_USER}"

# Generate sshd host keys if missing
ssh-keygen -A

# Make /workspace usable by the host user
chown "${HOST_USER}:${GROUP_NAME}" /workspace 2>/dev/null || true

# /etc/profile.d for interactive login shells
cat >/etc/profile.d/gpu-cosplay.sh <<EFP
export GPU_COSPLAY_CARD="${GPU_COSPLAY_CARD:-}"
export GPU_COSPLAY_PRETTY="${GPU_COSPLAY_PRETTY:-}"
export GPU_COSPLAY_VRAM_GB="${GPU_COSPLAY_VRAM_GB:-}"
export GPU_COSPLAY_FP32_TFLOPS="${GPU_COSPLAY_FP32_TFLOPS:-}"
export GPU_COSPLAY_BF16_TC_TFLOPS="${GPU_COSPLAY_BF16_TC_TFLOPS:-}"
export GPU_COSPLAY_BW_GBS="${GPU_COSPLAY_BW_GBS:-}"
export PIP_BREAK_SYSTEM_PACKAGES=1
EFP
chmod 644 /etc/profile.d/gpu-cosplay.sh

# /etc/environment for sshd non-interactive sessions (which skip /etc/profile.d)
{
    echo "GPU_COSPLAY_CARD=\"${GPU_COSPLAY_CARD:-}\""
    echo "GPU_COSPLAY_PRETTY=\"${GPU_COSPLAY_PRETTY:-}\""
    echo "GPU_COSPLAY_VRAM_GB=\"${GPU_COSPLAY_VRAM_GB:-}\""
    echo "GPU_COSPLAY_FP32_TFLOPS=\"${GPU_COSPLAY_FP32_TFLOPS:-}\""
    echo "GPU_COSPLAY_BF16_TC_TFLOPS=\"${GPU_COSPLAY_BF16_TC_TFLOPS:-}\""
    echo "GPU_COSPLAY_BW_GBS=\"${GPU_COSPLAY_BW_GBS:-}\""
    echo "PIP_BREAK_SYSTEM_PACKAGES=1"
    echo "LANG=C.UTF-8"
    echo "LC_ALL=C.UTF-8"
} >>/etc/environment

# Also enable sshd to pass GPU_COSPLAY_* through if client requests them
echo "AcceptEnv GPU_COSPLAY_* LANG LC_*" >>/etc/ssh/sshd_config

# Print a welcome message (lands in `docker logs`)
echo "================================================================"
echo "  gpu-cosplay container ready"
echo "  Cosplaying as: ${GPU_COSPLAY_PRETTY:-?}  (${GPU_COSPLAY_CARD:-?})"
echo "  VRAM cap:      ${GPU_COSPLAY_VRAM_GB:-?} GB"
echo "  User:          ${HOST_USER}  (uid=${HOST_UID} gid=${HOST_GID})"
echo "================================================================"

exec "$@"
