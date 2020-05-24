#!/bin/bash -eEu
# Performs dependency installation for Ubuntu/Debian systems, installs the Lua
# toolchain and installs luarocks necessary for the trashnet submodule, and
# establishes a Python3 virtual environment and installs requirements.

CMAKE='cmake'
CUDA='nvidia-cuda-toolkit'
GCC_ALT='gcc-4.8'
GPLUSPLUS_ALT='g++-4.8'
PIP3='pip3'
PYTHON3='python3.6'

function usage() {
    echo "usage: $(basename $0)"
}

if [[ ! "$#" -eq 0 ]]; then
    echo "ERROR: Invalid argument specified."
    usage
    exit 1
fi

function install_dependences() {
    sudo apt install -y "${PYTHON3}"
    sudo curl https://bootstrap.pypa.io/get-pip.py | "${PYTHON3}"

    sudo apt install -y "${CMAKE}"
    sudo apt install -y "${CUDA}"

    # Need gcc versions < 6 in order to install the cutorch rock. See more at:
    # https://github.com/torch/distro/issues/219.
    sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
    apt update
    sudo apt install -y "${GCC_ALT}" "${GPLUSPLUS_ALT}"
    sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/"${GCC_ALT}" \
    60 --slave /usr/bin/g++ g++ /usr/bin/"${GPLUSPLUS_ALT}"
}

function install_lua() {
    # Install Torch. See more at: http://torch.ch/docs/getting-started.html
    # Note: Repository had to be forked and modified due to:
    # https://github.com/torch/torch7/issues/1146
    if [[ -d "${HOME}/torch" ]]; then
        rm -rf "${HOME}/torch"
    fi

    # Install Torch
    export TORCH_NVCC_FLAGS="-D__CUDA_NO_HALF_OPERATORS__"
    git clone https://github.com/Cam2337/distro.git ~/torch --recursive
    pushd ~/torch
    bash install-deps
    bash install.sh
    popd

    # Refresh environment variables
    source ~/.bashrc
    echo "Torch7 installed at ${HOME}/torch. Remove with rm -rf ~/torch."
    echo "$(luajit -v)"
    echo "$(luarocks --version)"

    # Install required rocks for trashnet
    luarocks install torch
    luarocks install nn
    luarocks install optim
    luarocks install image
    luarocks install gnuplot
    luarocks install cutorch
    luarocks install cunn
}

function install_python() {
    # Setup virtual environment
    sudo apt install -y python3-venv
    "${PYTHON3}" -m venv ./venv

    # Activate virtual environment and install requirements
    source ./venv/bin/activate
    "${PIP3}" install -r ./requirements.txt
}

function main() {
    install_dependences
    install_lua
    install_python
}

main "$@"
