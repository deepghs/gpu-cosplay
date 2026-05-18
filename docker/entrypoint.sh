#!/bin/bash
# gpu-cosplay container entrypoint.
#
# Works in two modes:
#   - cosplay-baked image: all tools we need are pre-installed.
#   - bring-your-own image: we get bind-mounted in. Skip any step whose
#     tool is missing (sshd, ssh-keygen, etc.) and surface a clear warning.
set -uo pipefail

HOST_USER="${HOST_USER:-ubuntu}"
HOST_UID="${HOST_UID:-1000}"
HOST_GID="${HOST_GID:-1000}"

have() { command -v "$1" >/dev/null 2>&1; }
warn() { echo "[gpu-cosplay] WARN: $*" >&2; }

# ---------------------------------------------------------------------------
# 1. User and group: try to match the host's identity.
# ---------------------------------------------------------------------------
if have groupadd; then
    if ! getent group "${HOST_GID}" >/dev/null; then
        groupadd -g "${HOST_GID}" "${HOST_USER}" 2>/dev/null \
            || groupadd -g "${HOST_GID}" "cosplay-${HOST_GID}" 2>/dev/null \
            || warn "could not create group gid=${HOST_GID}"
    fi
fi
GROUP_NAME="$(getent group "${HOST_GID}" 2>/dev/null | cut -d: -f1)"
GROUP_NAME="${GROUP_NAME:-${HOST_USER}}"

if have useradd && ! id -u "${HOST_USER}" >/dev/null 2>&1; then
    if getent passwd "${HOST_UID}" >/dev/null; then
        existing="$(getent passwd "${HOST_UID}" | cut -d: -f1)"
        usermod -l "${HOST_USER}" -d "/home/${HOST_USER}" -m "${existing}" 2>/dev/null || true
        groupmod -n "${HOST_USER}" "$(id -gn "${HOST_USER}" 2>/dev/null)" 2>/dev/null || true
    else
        useradd -m -u "${HOST_UID}" -g "${HOST_GID}" -s /bin/bash "${HOST_USER}" 2>/dev/null \
            || warn "could not create user ${HOST_USER}"
    fi
fi
mkdir -p "/home/${HOST_USER}" 2>/dev/null || true

# ---------------------------------------------------------------------------
# 2. Sudo (optional — only if sudo is available).
# ---------------------------------------------------------------------------
if [[ -d /etc/sudoers.d ]] && have sudo; then
    echo "${HOST_USER} ALL=(ALL) NOPASSWD: ALL" >/etc/sudoers.d/90-cosplay 2>/dev/null \
        && chmod 440 /etc/sudoers.d/90-cosplay 2>/dev/null
fi

# ---------------------------------------------------------------------------
# 3. SSH key install + host keys (only if sshd is in the image).
# ---------------------------------------------------------------------------
SSH_DIR="/home/${HOST_USER}/.ssh"
mkdir -p "${SSH_DIR}"
chmod 700 "${SSH_DIR}"
if [[ -n "${GPU_COSPLAY_PUBKEY:-}" ]]; then
    echo "${GPU_COSPLAY_PUBKEY}" > "${SSH_DIR}/authorized_keys"
    chmod 600 "${SSH_DIR}/authorized_keys"
fi
chown -R "${HOST_UID}:${HOST_GID}" "${SSH_DIR}" "/home/${HOST_USER}" 2>/dev/null || true

HAVE_SSHD=0
if have sshd; then
    HAVE_SSHD=1
    mkdir -p /var/run/sshd 2>/dev/null || true
    have ssh-keygen && ssh-keygen -A 2>/dev/null || warn "ssh-keygen missing; using existing host keys"
    if [[ -f /etc/ssh/sshd_config ]]; then
        sed -i 's/#\?PermitRootLogin .*/PermitRootLogin no/' /etc/ssh/sshd_config 2>/dev/null || true
        sed -i 's/#\?PasswordAuthentication .*/PasswordAuthentication no/' /etc/ssh/sshd_config 2>/dev/null || true
        sed -i 's/#\?PubkeyAuthentication .*/PubkeyAuthentication yes/' /etc/ssh/sshd_config 2>/dev/null || true
        grep -q "^AcceptEnv GPU_COSPLAY_" /etc/ssh/sshd_config 2>/dev/null \
            || echo "AcceptEnv GPU_COSPLAY_* LANG LC_*" >>/etc/ssh/sshd_config
    fi
fi

# ---------------------------------------------------------------------------
# 4. /workspace ownership.
# ---------------------------------------------------------------------------
chown "${HOST_UID}:${HOST_GID}" /workspace 2>/dev/null || true

# ---------------------------------------------------------------------------
# 5. Env bake: profile.d for login shells, /etc/environment for sshd shells.
# ---------------------------------------------------------------------------
if [[ -d /etc/profile.d ]]; then
    cat >/etc/profile.d/gpu-cosplay.sh <<EFP
export GPU_COSPLAY_CARD="${GPU_COSPLAY_CARD:-}"
export GPU_COSPLAY_PRETTY="${GPU_COSPLAY_PRETTY:-}"
export GPU_COSPLAY_VRAM_GB="${GPU_COSPLAY_VRAM_GB:-}"
export GPU_COSPLAY_FP32_TFLOPS="${GPU_COSPLAY_FP32_TFLOPS:-}"
export GPU_COSPLAY_BF16_TC_TFLOPS="${GPU_COSPLAY_BF16_TC_TFLOPS:-}"
export GPU_COSPLAY_BW_GBS="${GPU_COSPLAY_BW_GBS:-}"
export PIP_BREAK_SYSTEM_PACKAGES=1
export PYTHONPATH="/opt/gpu-cosplay/python:\${PYTHONPATH:-}"
EFP
    chmod 644 /etc/profile.d/gpu-cosplay.sh
fi

if [[ -f /etc/environment ]] || touch /etc/environment 2>/dev/null; then
    # Avoid duplicating entries if the entrypoint reruns.
    sed -i '/^GPU_COSPLAY_/d;/^PIP_BREAK_SYSTEM_PACKAGES=/d;/^PYTHONPATH=/d' /etc/environment 2>/dev/null || true
    {
        echo "GPU_COSPLAY_CARD=\"${GPU_COSPLAY_CARD:-}\""
        echo "GPU_COSPLAY_PRETTY=\"${GPU_COSPLAY_PRETTY:-}\""
        echo "GPU_COSPLAY_VRAM_GB=\"${GPU_COSPLAY_VRAM_GB:-}\""
        echo "GPU_COSPLAY_FP32_TFLOPS=\"${GPU_COSPLAY_FP32_TFLOPS:-}\""
        echo "GPU_COSPLAY_BF16_TC_TFLOPS=\"${GPU_COSPLAY_BF16_TC_TFLOPS:-}\""
        echo "GPU_COSPLAY_BW_GBS=\"${GPU_COSPLAY_BW_GBS:-}\""
        echo "PIP_BREAK_SYSTEM_PACKAGES=1"
        echo "PYTHONPATH=/opt/gpu-cosplay/python"
    } >>/etc/environment
fi

# ---------------------------------------------------------------------------
# 6. Welcome banner.
# ---------------------------------------------------------------------------
echo "================================================================"
echo "  gpu-cosplay container ready"
echo "  Cosplaying as: ${GPU_COSPLAY_PRETTY:-?}  (${GPU_COSPLAY_CARD:-?})"
echo "  VRAM cap:      ${GPU_COSPLAY_VRAM_GB:-?} GB"
echo "  User:          ${HOST_USER}  (uid=${HOST_UID} gid=${HOST_GID})"
echo "  sshd:          $([[ ${HAVE_SSHD} = 1 ]] && echo present || echo NOT in image - use 'gpu-cosplay exec' instead)"
echo "================================================================"

# ---------------------------------------------------------------------------
# 7. Run the CMD. If the CMD asks for sshd and sshd is missing, fall back to
#    sleep so the container stays up and `docker exec` works.
# ---------------------------------------------------------------------------
if [[ $# -eq 0 ]]; then
    set -- sleep infinity
fi
if [[ "$1" == "/usr/sbin/sshd" || "$1" == "sshd" ]] && [[ ${HAVE_SSHD} = 0 ]]; then
    warn "CMD asks for sshd but it isn't in this image. Staying alive with 'sleep infinity'."
    warn "Use 'gpu-cosplay exec <name> -- <cmd>' to run commands in this container."
    exec sleep infinity
fi
exec "$@"
