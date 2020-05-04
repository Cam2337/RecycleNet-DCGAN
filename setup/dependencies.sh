#!/bin/bash -eEu
# Installs necessary dependent packages for Ubuntu/Debian systems.

CMAKE='cmake'
CUDA='nvidia-cuda-toolkit'
GCC_ALT='gcc-4.8'
GPLUSPLUS_ALT='g++-4.8'
PYTHON3='python3.6'

function usage() {
    echo "usage: sudo $(basename $0)"
}

if [[ ! "${EUID}" -eq 0 ]]; then
    echo "ERROR: $(basename $0) must be run with sudo privileges."
    usage
    exit 1
fi

if [[ ! "$#" -eq 0 ]]; then
    echo "ERROR: Invalid argument specified."
    usage
    exit 1
fi

function main() {
    apt install "${PYTHON3}"
    curl https://bootstrap.pypa.io/get-pip.py | "${PYTHON3}"

    apt install "${CMAKE}"
    apt install "${CUDA}"

    # Need gcc versions < 6 in order to install the cutorch rock. See more at:
    # https://github.com/torch/distro/issues/219.
    add-apt-repository ppa:ubuntu-toolchain-r/test
    apt update
    apt install "${GCC_ALT}" "${GPLUSPLUS_ALT}"
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/"${GCC_ALT}" 60 \
    --slave /usr/bin/g++ g++ /usr/bin/"${GPLUSPLUS_ALT}"
}

main "$@"
