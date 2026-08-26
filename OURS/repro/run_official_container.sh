#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
image="${BLB_IMAGE:-crpi-hxmgk7l66ku6bx2x.cn-hangzhou.personal.cr.aliyuncs.com/tianshixu/ppdl:blbv1}"
name="${BLB_CONTAINER_NAME:-blb_formal_env}"
host_uid="$(id -u)"
host_gid="$(id -g)"
container_user="${BLB_CONTAINER_USER:-blbhost}"
cuda_root="${CUDA_ROOT:-${CUDAToolkit_ROOT:-}}"

# The official image is intentionally used only as an environment.  It does
# not contain nvcc, so expose a host CUDA toolkit read-only.  Prefer an
# explicit CUDA_ROOT, then the nvcc already on PATH, then the conventional
# /usr/local/cuda symlink, then the newest versioned toolkit under /usr/local.
if [[ -z "$cuda_root" ]]; then
  if command -v nvcc >/dev/null 2>&1; then
    cuda_compiler="$(readlink -f "$(command -v nvcc)")"
    cuda_root="$(cd "$(dirname "$cuda_compiler")/.." && pwd -P)"
  elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
    cuda_root="$(readlink -f /usr/local/cuda)"
  else
    cuda_compiler="$(find /usr/local -maxdepth 4 -type f -path '*/bin/nvcc' -perm -111 -print 2>/dev/null | sort -V | tail -n 1 || true)"
    if [[ -n "$cuda_compiler" ]]; then
      cuda_root="$(cd "$(dirname "$cuda_compiler")/.." && pwd -P)"
    fi
  fi
fi

cuda_args=()
if [[ -n "$cuda_root" && -x "$cuda_root/bin/nvcc" ]]; then
  cuda_ld_path="$cuda_root/lib64:$cuda_root/targets/x86_64-linux/lib"
  if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    cuda_ld_path="$cuda_ld_path:$LD_LIBRARY_PATH"
  fi
  cuda_args+=(
    --env "CUDA_HOME=$cuda_root"
    --env "CUDA_PATH=$cuda_root"
    --env "CUDAToolkit_ROOT=$cuda_root"
    --env "PATH=$cuda_root/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    --env "LD_LIBRARY_PATH=$cuda_ld_path"
    --mount "type=bind,src=$cuda_root,dst=$cuda_root,readonly"
  )
else
  echo "BLB_CONTAINER status=BLOCKED reason=host_nvcc_missing" >&2
  echo "Set CUDA_ROOT to a host toolkit containing bin/nvcc, for example CUDA_ROOT=/usr/local/cuda-12.3." >&2
  exit 2
fi

if ! docker info >/dev/null 2>&1; then
  echo "BLB_CONTAINER status=BLOCKED reason=docker_daemon_permission" >&2
  exit 2
fi
if [[ "${PULL_IMAGE:-0}" == "1" ]]; then
  docker pull "$image"
fi
if ! docker image inspect "$image" >/dev/null 2>&1; then
  echo "BLB_CONTAINER status=BLOCKED reason=image_missing image=$image" >&2
  echo "Set PULL_IMAGE=1, or load the verified Zenodo tar first." >&2
  exit 2
fi

echo "BLB_CONTAINER source_root=$repo image=$image"
echo "BLB_CONTAINER policy=container_environment_only internal_/home/CKKS-MPC_not_used"

# Reuse a container created by an earlier invocation. This avoids Docker's
# name-conflict error after a terminal was closed without exiting the shell.
# Fail closed if the name belongs to another image or another source mount.
existing_id="$(docker container ls -aq --filter "name=^/${name}$")"
if [[ -n "$existing_id" ]]; then
  existing_image="$(docker container inspect --format '{{.Config.Image}}' "$existing_id")"
  existing_source="$(docker container inspect --format '{{range .Mounts}}{{if eq .Destination "/workspace/BLB"}}{{.Source}}{{end}}{{end}}' "$existing_id")"
  if [[ "$existing_image" != "$image" || "$existing_source" != "$repo" ]]; then
    echo "BLB_CONTAINER status=BLOCKED reason=existing_container_mismatch name=$name" >&2
    echo "Use a different BLB_CONTAINER_NAME, or inspect and remove the old container yourself." >&2
    exit 2
  fi

  existing_running="$(docker container inspect --format '{{.State.Running}}' "$existing_id")"
  if [[ "$existing_running" != "true" ]]; then
    docker start "$existing_id" >/dev/null
  fi
  echo "BLB_CONTAINER status=REUSE name=$name id=$existing_id"
  exec docker exec -it \
    --user "$host_uid:$host_gid" \
    --env HOME=/tmp \
    --workdir /workspace/BLB \
    "$existing_id" /bin/bash
fi

# The image has no entry for arbitrary host GIDs (for example GID 136).
# Start as root only long enough to create a transient passwd/group entry;
# the interactive shell and all BLB builds then run as the host UID/GID.
exec docker run --rm --gpus all --name "$name" --cap-add=NET_ADMIN \
  --shm-size=16g \
  --user 0:0 \
  --env HOME=/tmp \
  --env "BLB_HOST_UID=$host_uid" \
  --env "BLB_HOST_GID=$host_gid" \
  --env "BLB_CONTAINER_USER=$container_user" \
  "${cuda_args[@]}" \
  --mount "type=bind,src=$repo,dst=/workspace/BLB" \
  --workdir /workspace/BLB \
  -it "$image" /bin/bash -lc '
    set -e
    uid="${BLB_HOST_UID:?}"
    gid="${BLB_HOST_GID:?}"
    user_name="${BLB_CONTAINER_USER:-blbhost}"
    group_name="blbhostgrp"

    if getent group "$gid" >/dev/null 2>&1; then
      group_name="$(getent group "$gid" | cut -d: -f1)"
    else
      groupadd --gid "$gid" "$group_name"
    fi

    if getent passwd "$uid" >/dev/null 2>&1; then
      user_name="$(getent passwd "$uid" | cut -d: -f1)"
    elif ! getent passwd "$user_name" >/dev/null 2>&1; then
      useradd --uid "$uid" --gid "$gid" --home-dir /tmp --shell /bin/bash --no-create-home "$user_name"
    fi

    cd /workspace/BLB
    exec runuser --preserve-environment --user "$user_name" --group "$group_name" -- /bin/bash
  '
