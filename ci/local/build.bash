#! /usr/bin/env bash

# Copyright (c) 2018-2020 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Released under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.

################################################################################
# Thrust and CUB local containerized build script
################################################################################

function usage {
  echo "Usage: ${0} [flags...] [cmake-targets...]"
  echo
  echo "Build and test your local repository using a gpuCI Docker image."
  echo "If CMake targets are specified, only those targets are built and tested."
  echo "Otherwise, everything is built and tested."
  echo
  echo "-h, -help, --help"
  echo "  Print this message."
  echo
  echo "-r <path>, --repository <path>"
  echo "  Path to the repository (default: ${REPOSITORY_PATH})."
  echo
  echo "-i <image>, --image <image>"
  echo "  Docker image to use (default: ${IMAGE})"
  echo
  echo "-s, --shell-only"
  echo "  Skip building and testing and launch an interactive shell instead."

  exit -3
}

SCRIPT_PATH=$(cd $(dirname ${0}); pwd -P)

REPOSITORY_PATH=$(realpath ${SCRIPT_PATH}/../..)

################################################################################
# FLAGS - Process command line flags.
################################################################################

IMAGE="gpuci/cccl:cuda11.0-devel-ubuntu18.04-gcc5"

SHELL_ONLY=0

TARGETS=""

while test ${#} != 0
do
  case "${1}" in
  -h) ;&
  -help) ;&
  --help) usage ;;
  -r) ;&
  --repository)
    shift # The next argument is the path.
    REPOSITORY_PATH="${1}"
    ;;
  -i) ;&
  --image)
    shift # The next argument is the image.
    IMAGE="${1}"
    ;;
  -s) ;&
  --shell-only) SHELL_ONLY=1 ;;
  *)
    TARGETS="${TARGETS:+${TARGETS} }${1}"
    ;;
  esac
  shift
done

################################################################################
# PATHS - Setup paths for the container.
################################################################################

# ${REPOSITORY_PATH} is the local filesystem path to the Git repository being
# built and tested. It can be set with the --repository flag.
#
# ${BUILD_PATH} is the local filesystem path that will be used for the build. It
# is named after the image name, allowing multiple image builds to coexist on
# the local filesystem.
#
# ${REPOSITORY_PATH_IN_CONTAINER} is the location of ${REPOSITORY_PATH} inside
# the container.
#
# ${BUILD_PATH_IN_CONTAINER} is the location of ${BUILD_PATH} inside the
# container.

BUILD_PATH=${REPOSITORY_PATH}/build_$(echo "$(basename "${IMAGE}")" | sed -e 's/:/_/g')
mkdir -p ${BUILD_PATH}

BASE_PATH_IN_CONTAINER="/cccl"

REPOSITORY_PATH_IN_CONTAINER="${BASE_PATH_IN_CONTAINER}/$(basename "${REPOSITORY_PATH}")"

BUILD_PATH_IN_CONTAINER="${BASE_PATH_IN_CONTAINER}/$(basename "${REPOSITORY_PATH}")/build"

################################################################################
# COMMAND - Setup the command that will be run by the container.
################################################################################

if   [ "${SHELL_ONLY}" != 0 ]; then
  COMMAND="bash"
else
  COMMAND="${REPOSITORY_PATH_IN_CONTAINER}/ci/cpu/build.bash ${TARGETS} || bash"
fi

################################################################################
# PERMISSIONS - Setup permissions and users for hte container.
################################################################################

PASSWD_PATH="/etc/passwd"
GROUP_PATH="/etc/group"

USER_FOUND=$(grep -wc "$(whoami)" < "${PASSWD_PATH}")
if [ "${USER_FOUND}" == 0 ]; then
  echo "Local user not found, generating dummy /etc/passwd and /etc/group."
  cp "${PASSWD_PATH}" /tmp/passwd
  PASSWD_PATH="/tmp/passwd"
  cp "${GROUP_PATH}" /tmp/group
  GROUP_PATH="/tmp/group"
  echo "$(whoami):x:$(id -u):$(id -g):$(whoami),,,:${HOME}:${SHELL_ONLY}" >> "${PASSWD_PATH}"
  echo "$(whoami):x:$(id -g):" >> "${GROUP_PATH}"
fi

################################################################################
# GPU - Setup GPUs.
################################################################################

# Limit GPUs available to the container based on ${CUDA_VISIBLE_DEVICES}.
if [ -z "${CUDA_VISIBLE_DEVICES}" ]; then
  VISIBLE_DEVICES="all"
else
  VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
fi

DOCKER_MAJOR_VER=$(docker -v | sed 's/[^[0-9]*\([0-9]*\).*/\1/')
GPU_OPTS="--gpus device=${VISIBLE_DEVICES}"
if [ "${DOCKER_MAJOR_VER}" -lt 19 ]
then
  GPU_OPTS="--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES='${VISIBLE_DEVICES}'"
fi

################################################################################
# LAUNCH - Pull and launch the container.
################################################################################

NVIDIA_DOCKER_INSTALLED=$(docker info 2>&1 | grep -i runtime | grep -c nvidia)
if [ "${NVIDIA_DOCKER_INSTALLED}" == 0 ]; then
  echo "NVIDIA Docker not found, please install it: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#installing-docker-ce"
  exit -4
fi

#docker pull "${IMAGE}"

docker run --rm -it ${GPU_OPTS} \
  --cap-add=SYS_PTRACE \
  --user "$(id -u)":"$(id -g)" \
  -v "${REPOSITORY_PATH}":"${REPOSITORY_PATH_IN_CONTAINER}" \
  -v "${BUILD_PATH}":"${BUILD_PATH_IN_CONTAINER}" \
  -v "${PASSWD_PATH}":/etc/passwd:ro \
  -v "${GROUP_PATH}":/etc/group:ro \
  -e "WORKSPACE=${REPOSITORY_PATH_IN_CONTAINER}" \
  -w "${REPOSITORY_PATH_IN_CONTAINER}" \
  "${IMAGE}" bash -c "${COMMAND}"

