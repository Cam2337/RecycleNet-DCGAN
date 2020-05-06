#!/bin/bash -eEu
# Establishes a Python virtual environment and installs requirements.

PIP3='pip3'
PYTHON3='python3.6'

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
    # Setup virtual environment
    sudo apt-get install python3-venv
    "${PYTHON3}" -m venv ./venv

    # Activate virtual environment and install requirements
    source ./venv/bin/activate
    "${PIP3}" install -r ./requirements.txt
}

main "$@"
