#!/bin/bash

set -e

docker run \
  --privileged \
  --gpus=all \
  -v /dev:/dev \
  -v /tm.X11-unix/:/tmp/.X11-unix \
  -v /tmp/.docker.xauth:/tmp/.docker.xauth \
  -v /home/tartanauv/tauv-mono/packages:/shared/tauv_ws/src/tauv-packages \
  -e XAUTHORITY=/tmp/.docker.xauth \
  --net=host \
  -it \
  tauv/kingfisher /bin/bash
