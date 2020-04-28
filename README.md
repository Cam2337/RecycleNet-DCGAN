# cs230-project
CS230: Deep Learning and Neural Networks Project

# Getting Started
## Linux - Ubuntu/Debian
In order to run this project, several requirements need to be met. First and foremost, we must install requirements necessary to run the [trashnet](https://github.com/garythung/trashnet) project, which relies on the JITLua toolchain and the [Torch](http://torch.ch/docs/getting-started.html) project. Once these preqrequisites are installed, we need to setup a Python3 virtual environment, and install packages necessary to run the GAN to generate synthetic training images. Both of these steps are preceded by installation of some basic development tools. The order of operations is as follows:

1. `sudo ./dependencies.sh` - This will install preqrequisite dependencies including `cmake`, `CUDA`, `GCC-4.8`, and `Python3.6`
2. `./lua_setup.sh` - This will install the [Torch](http://torch.ch/docs/getting-started.html) project, and setup the necessary Lua rocks for running trashnet
3. `./python_setup.sh` - This will setup the project's virtual environment in the project root directory, and install necessary Python3 packages
