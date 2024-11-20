#!/bin/bash

# Check if there are any running Docker containers
if [ "$(docker ps -q)" ]; then
    echo "There are running container. You can see them by running 'docker ps' and stop them by calling 'docker kill <container name>'. Aborting..."
    exit 1
fi

if [[ -z "${DOCKER_SHARED_DIR}" ]]; then
    mkdir -p $HOME/shared
    SHARED_DIR=$HOME/shared
else
    SHARED_DIR=$DOCKER_SHARED_DIR
fi

echo "Using shared folder: $SHARED_DIR"

HOSTNAME=$(hostname)

echo "Starting container, VNC address: http://${HOSTNAME}.lan.local.cmu.edu:60$(id -u | rev | cut -c1-3 | rev)/vnc.html"

SCRIPT_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
TAUV_MONO_DIR=$(realpath "$SCRIPT_DIR/../../")

rocker \
	--nvidia \
	--privileged \
	--port 60$(id -u | rev | cut -c1-3 | rev):8080 \
  --volume $TAUV_MONO_DIR/packages:/tauv-packages \
	--volume $SHARED_DIR:/shared \
	--volume /dev/shm:/dev/shm \
	-- tauv/x86-nvidia-workstation


