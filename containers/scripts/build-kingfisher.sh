set -e

CONTAINERS_DIR="$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]}")")")"

cd $CONTAINERS_DIR

docker build --file platform/Dockerfile.orin --tag tauv/orin-platform .
docker build --build-arg BASE_IMAGE=tauv/orin-platform --file common/Dockerfile.common --tag tauv/orin-common .
docker build --build-arg BASE_IMAGE=tauv/orin-common --file apps/vehicle/Dockerfile.kingfisher --tag tauv/kingfisher .

echo -e "\n\n\nFinished building Kingfisher image."
