#!/usr/bin/env bash
# The notebook image's entrypoint: copy the baked-in notebook folder into the writable work folder (when it is
# empty, as a new emptyDir is), then run JupyterLab there.
set -euo pipefail

src="${RHVP_NOTEBOOKS:-/opt/app-root/notebooks}"
work="${RHVP_WORK:-/work}"

mkdir -p "$work"
if [ -z "$(ls -A "$work")" ]; then
  cp -R "$src/." "$work/"
  # The baked-in copy is read-only; the working copy is the user's to edit. Only the copied files change: the
  # work folder itself may belong to root (an emptyDir does).
  find "$work" -mindepth 1 -exec chmod u+w {} +
fi
mkdir -p "${JUPYTER_RUNTIME_DIR:-/tmp/jupyter-runtime}"
# Jupyter and IPython keep their settings under HOME. With a read-only root file system, use one under /tmp.
if [ ! -w "${HOME:-/nonexistent}" ]; then
  export HOME=/tmp/home
  mkdir -p "$HOME"
fi

# No Jupyter token and no password: JupyterLab listens on 127.0.0.1 only, inside the pod. The way in is the pod's
# OpenShift oauth-proxy sidecar (port 8443, behind the reencrypt Route), which asks for an OpenShift login and is
# the authentication. allow_remote_access accepts the Route's host name, which the proxy passes on.
exec jupyter lab \
  --ServerApp.ip=127.0.0.1 \
  --ServerApp.port=8888 \
  --ServerApp.root_dir="$work" \
  --IdentityProvider.token='' \
  --ServerApp.allow_remote_access=True \
  --ServerApp.open_browser=False \
  "$@"
