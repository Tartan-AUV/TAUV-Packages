#!/bin/bash

set -e

if [ -n "$GITHUB_WORKSPACE" ]; then
  # If inside GitHub Actions, use GITHUB_WORKSPACE as the repo root
  REPO_ROOT="$GITHUB_WORKSPACE"
else
  REPO_ROOT=$(pwd)
fi

cd $REPO_ROOT/containers

COMMIT_HASH=$(git rev-parse --short HEAD)

docker build --file platform/Dockerfile.x86-nvidia --tag "tauv/x86-nvidia-platform:$COMMIT_HASH" .
docker build --build-arg BASE_IMAGE="tauv/x86-nvidia-platform:$COMMIT_HASH" --file common/Dockerfile.common --tag "tauv/x86-nvidia-common:$COMMIT_HASH" .
docker build --build-arg BASE_IMAGE="tauv/x86-nvidia-common:$COMMIT_HASH" --file apps/surface/Dockerfile.ci --tag "tauv/x86-nvidia-ci:$COMMIT_HASH" .

echo -e "\n\n\nFinished building the CI image: $COMMIT_HASH"
