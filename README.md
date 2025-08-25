[**English**](./README.md) | [中文](./README_tw.md)

# Capybara

**An Integrated Python Package for Image Processing and Deep Learning.**

<p align="left">
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202-dfd.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/python-3.10+-aff.svg"></a>
    <a href="https://github.com/DocsaidLab/Capybara/releases"><img src="https://img.shields.io/github/v/release/DocsaidLab/Capybara?color=ffa"></a>
    <a href="https://pypi.org/project/capybara_docsaid/"><img src="https://img.shields.io/pypi/v/capybara_docsaid.svg"></a>
    <a href="https://pypi.org/project/capybara_docsaid/"><img src="https://img.shields.io/pypi/dm/capybara_docsaid?color=9cf"></a>
</p>

## Introduction

![title](https://raw.githubusercontent.com/DocsaidLab/Capybara/refs/heads/main/docs/title.webp)

This project is an image processing and deep learning toolkit, mainly consisting of the following parts:

- **Vision**: Provides functionalities related to computer vision, such as image and video processing.
- **Structures**: Modules for handling structured data, such as BoundingBox and Polygon.
- **ONNXEngine**: Provides ONNX inference functionalities, supporting ONNX format models.
- **Utils**: Contains utility functions that do not belong to other modules.
- **Tests**: Includes test code for various functions to verify their correctness.

## Technical Documentation

For more detailed information on installation and usage, please refer to the [**Capybara Documents**](https://docsaid.org/en/docs/capybara).

The document provides a detailed explanation of this project and answers to frequently asked questions.

## Prerequisites

Before the installation of Capybara, ensure that your system meets the following requirements:

### Python Version

3.10+

### Dependency Packages

Please install the necessary system packages according to your operating system:

#### Ubuntu

```bash
sudo apt install libturbojpeg exiftool ffmpeg libheif-dev poppler-utils
```

##### GPU Dependencies

To use ONNX Runtime with GPU acceleration, ensure that you install a compatible version, which can be found on the official ONNX Runtime CUDA Execution Provider requirements page.

Here's an example to install cuda-12.8:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-8
# Post installation, add cuda path to .bashrc or .zshrc
export shellrc="~/.zshrc"
echo 'export PATH=/usr/local/cuda-12.8/bin${PATH:+:${PATH}}' >> $shellrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> $shellrc
```

For more details, please see [Nvidia CUDA](https://developer.nvidia.com/cuda-toolkit).

#### MacOS

```bash
brew install jpeg-turbo exiftool ffmpeg libheif poppler
```

## Installation

### PyPI

```bash
pip install capybara-docsaid
```

### Git

```bash
pip install git+https://github.com/DocsaidLab/Capybara.git
```

## Docker for Deployment

We provide a Docker script for convenient deployment, ensuring a consistent environment. Below are the steps to build the image with Capybara installed.

1. Clone this repository:

   ```bash
   git clone https://github.com/DocsaidLab/Capybara.git
   ```

2. Enter the project directory and run the build script:

   ```bash
   cd Capybara
   bash docker/build.bash
   ```

   This will build an image using the [**Dockerfile**](docker/Dockerfile) in the project. The image is based on `nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04` by default, providing the CUDA environment required for ONNXRuntime inference.

3. After the build is complete, mount the working directory and run the program:

   ```bash
   docker run --gpus all -it --rm capybara_docsaid:latest bash
   ```

**PS: If you want to compile cuda or cudnn for developing, please change the base image to `nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04`.**

### gosu Permissions Issues

If you encounter issues with file ownership as root when running scripts inside the container, causing permission problems, you can use `gosu` to switch users in the Dockerfile. Specify `USER_ID` and `GROUP_ID` when starting the container to avoid frequent permission adjustments in collaborative development.

For details, refer to the technical documentation: [**Integrating gosu Configuration**](https://docsaid.org/en/docs/capybara/advance/#integrating-gosu-configuration)

1. Install `gosu`:

   ```dockerfile
   RUN apt-get update && apt-get install -y gosu
   ```

2. Use `gosu` in the container start command to switch to a non-root user for file read/write operations.

   ```dockerfile
   # Create the entrypoint script
   RUN printf '#!/bin/bash\n\
       if [ ! -z "$USER_ID" ] && [ ! -z "$GROUP_ID" ]; then\n\
           groupadd -g "$GROUP_ID" -o usergroup\n\
           useradd --shell /bin/bash -u "$USER_ID" -g "$GROUP_ID" -o -c "" -m user\n\
           export HOME=/home/user\n\
           chown -R "$USER_ID":"$GROUP_ID" /home/user\n\
           chown -R "$USER_ID":"$GROUP_ID" /code\n\
       fi\n\
       \n\
       # Check for parameters\n\
       if [ $# -gt 0 ]; then\n\
           exec gosu ${USER_ID:-0}:${GROUP_ID:-0} python "$@"\n\
       else\n\
           exec gosu ${USER_ID:-0}:${GROUP_ID:-0} bash\n\
       fi' > "$ENTRYPOINT_SCRIPT"

   RUN chmod +x "$ENTRYPOINT_SCRIPT"

   ENTRYPOINT ["/bin/bash", "/entrypoint.sh"]
   ```

For more advanced configuration, refer to [**NVIDIA Container Toolkit**](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) and the official [**docker**](https://docs.docker.com/) documentation.

## Testing

This project uses `pytest` for unit testing, and users can run the tests themselves to verify the correctness of the functionalities. To install and run the tests, use the following commands:

```bash
pip install pytest
python -m pytest -vv tests
```

Once completed, you can check if all modules are functioning properly. If any issues arise, first check the environment settings and package versions.

If the problem persists, please report it in the Issue section.

## Citation

```bibtex
@misc{lin2025capybara,
  author       = {Kun-Hsiang Lin*, Ze Yuan*},
  title        = {Capybara: An Integrated Python Package for Image Processing and Deep Learning.},
  year         = {2025},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/DocsaidLab/Capybara}},
  note         = {* equal contribution}
}
```
