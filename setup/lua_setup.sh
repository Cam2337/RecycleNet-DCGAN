#!/bin/bash -eEu
# Sets up Lua installation necessary for trashnet submodule.

function usage() {
    echo "usage: $(basename $0)"
}

if [[ "${EUID}" -eq 0 ]]; then
    echo "ERROR: $(basename $0) must be run *wihtout* with sudo privileges."
    usage
    exit 1
fi

if [[ ! "$#" -eq 0 ]]; then
    echo "ERROR: Invalid argument specified."
    usage
    exit 1
fi

function main() {
    # Install Torch. See more at: http://torch.ch/docs/getting-started.html
    # Note: Repository had to be forked and modified due to https://github.com/torch/torch7/issues/1146
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

main "$@"
