#!/bin/bash
# gpu-cosplay container entrypoint.
#
# Works both for the baked image and for bring-your-own (BYO) images that get
# bind-mounted with our runtime. Each step is guarded so missing tools surface
# a clear warning instead of crashing the container.
set -uo pipefail

HOST_USER="${HOST_USER:-ubuntu}"
HOST_UID="${HOST_UID:-1000}"
HOST_GID="${HOST_GID:-1000}"

have() { command -v "$1" >/dev/null 2>&1; }
warn() { echo "[gpu-cosplay] WARN: $*" >&2; }

# ---------------------------------------------------------------------------
# 1. Match the host's user identity.
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
# 2. Sudo NOPASSWD for convenience.
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
    have ssh-keygen && ssh-keygen -A 2>/dev/null || warn "ssh-keygen missing"
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
# 5. Bake env vars so they're visible from any shell (login or non-login,
#    sshd-launched or docker-exec'd).
# ---------------------------------------------------------------------------
if [[ -d /etc/profile.d ]]; then
    cat >/etc/profile.d/gpu-cosplay.sh <<EFP
export GPU_COSPLAY_CARD="${GPU_COSPLAY_CARD:-}"
export GPU_COSPLAY_PRETTY="${GPU_COSPLAY_PRETTY:-}"
export GPU_COSPLAY_VRAM_GB="${GPU_COSPLAY_VRAM_GB:-}"
export GPU_COSPLAY_FP32_TFLOPS="${GPU_COSPLAY_FP32_TFLOPS:-}"
export GPU_COSPLAY_BF16_TC_TFLOPS="${GPU_COSPLAY_BF16_TC_TFLOPS:-}"
export GPU_COSPLAY_BW_GBS="${GPU_COSPLAY_BW_GBS:-}"
export GPU_COSPLAY_TDP_W="${GPU_COSPLAY_TDP_W:-}"
export GPU_COSPLAY_PHYS_VRAM_MIB="${GPU_COSPLAY_PHYS_VRAM_MIB:-}"
export PIP_BREAK_SYSTEM_PACKAGES=1
EFP
    chmod 644 /etc/profile.d/gpu-cosplay.sh
fi

if [[ -f /etc/environment ]] || touch /etc/environment 2>/dev/null; then
    sed -i '/^GPU_COSPLAY_/d;/^PIP_BREAK_SYSTEM_PACKAGES=/d' /etc/environment 2>/dev/null || true
    {
        echo "GPU_COSPLAY_CARD=\"${GPU_COSPLAY_CARD:-}\""
        echo "GPU_COSPLAY_PRETTY=\"${GPU_COSPLAY_PRETTY:-}\""
        echo "GPU_COSPLAY_VRAM_GB=\"${GPU_COSPLAY_VRAM_GB:-}\""
        echo "GPU_COSPLAY_FP32_TFLOPS=\"${GPU_COSPLAY_FP32_TFLOPS:-}\""
        echo "GPU_COSPLAY_BF16_TC_TFLOPS=\"${GPU_COSPLAY_BF16_TC_TFLOPS:-}\""
        echo "GPU_COSPLAY_BW_GBS=\"${GPU_COSPLAY_BW_GBS:-}\""
        echo "GPU_COSPLAY_TDP_W=\"${GPU_COSPLAY_TDP_W:-}\""
        echo "GPU_COSPLAY_PHYS_VRAM_MIB=\"${GPU_COSPLAY_PHYS_VRAM_MIB:-}\""
        echo "PIP_BREAK_SYSTEM_PACKAGES=1"
    } >>/etc/environment
fi

# ---------------------------------------------------------------------------
# 6. Install the Python runtime hook (.pth + module) into site-packages.
#    This makes monkey-patching automatic — user code does NOT need to
#    `import gpu_cosplay_inject` (which is gone). Works for whatever python3
#    is on PATH (system, /opt/conda, venv, etc.).
# ---------------------------------------------------------------------------
RUNTIME_SRC=/opt/gpu-cosplay/python/gpu_cosplay_runtime.py
PTH_SRC=/opt/gpu-cosplay/python/gpu_cosplay_runtime.pth
if [[ -f "${RUNTIME_SRC}" ]] && have python3; then
    SITE_DIR="$(python3 -c 'import site,sys; print(site.getsitepackages()[0])' 2>/dev/null)"
    if [[ -n "${SITE_DIR}" ]] && [[ -d "${SITE_DIR}" || $(mkdir -p "${SITE_DIR}" 2>/dev/null; echo $?) -eq 0 ]]; then
        ln -sf "${RUNTIME_SRC}" "${SITE_DIR}/gpu_cosplay_runtime.py" 2>/dev/null \
            || cp "${RUNTIME_SRC}" "${SITE_DIR}/gpu_cosplay_runtime.py" 2>/dev/null
        ln -sf "${PTH_SRC}"     "${SITE_DIR}/gpu_cosplay_runtime.pth" 2>/dev/null \
            || cp "${PTH_SRC}"     "${SITE_DIR}/gpu_cosplay_runtime.pth" 2>/dev/null
    else
        warn "could not locate python3 site-packages; torch/pynvml monkey-patches won't auto-load."
        warn "Workaround: set PYTHONPATH=/opt/gpu-cosplay/python and import gpu_cosplay_runtime manually."
    fi
fi

# ---------------------------------------------------------------------------
# 7. Install nvidia-smi shim + verify tool when bind-mounted in (BYO mode).
#    The shim at /usr/local/bin/nvidia-smi shadows /usr/bin/nvidia-smi because
#    /usr/local/bin precedes /usr/bin on PATH. We never touch /usr/bin/nvidia-smi.
# ---------------------------------------------------------------------------
if [[ -x /opt/gpu-cosplay/nvidia-smi ]] && [[ ! -x /usr/local/bin/nvidia-smi ]]; then
    ln -sf /opt/gpu-cosplay/nvidia-smi /usr/local/bin/nvidia-smi 2>/dev/null \
        || cp /opt/gpu-cosplay/nvidia-smi /usr/local/bin/nvidia-smi 2>/dev/null
fi
if [[ -x /opt/gpu-cosplay/gpu_cosplay_verify.py ]] && [[ ! -x /usr/local/bin/gpu-cosplay-verify ]]; then
    ln -sf /opt/gpu-cosplay/gpu_cosplay_verify.py /usr/local/bin/gpu-cosplay-verify 2>/dev/null \
        || cp /opt/gpu-cosplay/gpu_cosplay_verify.py /usr/local/bin/gpu-cosplay-verify 2>/dev/null
fi

# ---------------------------------------------------------------------------
# 8. Welcome banner.
# ---------------------------------------------------------------------------
echo "================================================================"
echo "  gpu-cosplay container ready"
echo "  Cosplaying as: ${GPU_COSPLAY_PRETTY:-?}  (${GPU_COSPLAY_CARD:-?})"
echo "  VRAM cap:      ${GPU_COSPLAY_VRAM_GB:-?} GB"
echo "  User:          ${HOST_USER}  (uid=${HOST_UID} gid=${HOST_GID})"
echo "  sshd:          $([[ ${HAVE_SSHD} = 1 ]] && echo present || echo NOT in image - use 'gpu-cosplay exec')"
echo "================================================================"

# ---------------------------------------------------------------------------
# 9. Run the CMD. Fall back to `sleep infinity` if sshd was requested but
#    isn't in the image — that way `docker exec` still works.
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
