[English](./README.md) | **[中文](./README_tw.md)**

# Capybara

<p align="left">
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202-dfd.svg"></a>
    <a href="https://github.com/DocsaidLab/Capybara/releases"><img src="https://img.shields.io/github/v/release/DocsaidLab/Capybara?color=ffa"></a>
    <a href=""><img src="https://img.shields.io/badge/python-3.8+-aff.svg"></a>
</p>

## Introduction

![title](./docs/title.webp)

This project is a toolbox for image processing and deep learning, primarily consisting of the following components:

- **Vision**: Functions related to computer vision, such as image and video processing.
- **Structures**: Modules for handling structured data, such as BoundingBox and Polygon.
- **ONNXEngine**: Provides ONNX inference capabilities, supporting ONNX format models.
- **Utils**: Miscellaneous utilities that do not fit into other categories.
- **Tests**: Test files for verifying the functionality of various functions.

## Documentation

For installation and usage instructions, please refer to the [**Capybara Documents**](https://docsaid.org/en/docs/capybara).

Here, you will find all the detailed information about this project.

## Installation

Before installing Capybara, ensure your system meets the following requirements:

### Python Version

- Ensure Python 3.10 or higher is installed on your system.

### Dependencies

Install the required dependencies based on your operating system.

- **Ubuntu**

  Open the terminal and run the following commands to install dependencies:

  ```bash
  sudo apt install libturbojpeg exiftool ffmpeg libheif-dev
  ```

- **MacOS**

  Use brew to install dependencies:

  ```bash
  brew install jpeg-turbo exiftool ffmpeg libheif
  ```

### pdf2image Dependencies

pdf2image is a Python module for converting PDF documents into images.

Follow these instructions to install it based on your operating system:

- For detailed installation instructions, refer to the [**pdf2image**](https://github.com/Belval/pdf2image) project page.

- MacOS: Mac users need to install poppler. Install it via Brew:

  ```bash
  brew install poppler
  ```

- Linux: Most Linux distributions come with `pdftoppm` and `pdftocairo` pre-installed.

  If not, install poppler-utils via your package manager:

  ```bash
  sudo apt install poppler-utils
  ```

### Onnxruntime GPU Dependencies

- CUDA 12.4

  ```bash
  sudo apt install cuda-12-4
  rc_file = ~/.bashrc # (or to your rc_file such as ~/.zshrc)
  echo 'export PATH=/usr/local/cuda-12.4/bin${PATH:+:${PATH}}' >> $rc_file # Add CUDA to PATH
  echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> $rc_file # Add CUDA library to LD_LIBRARY_PATH
  ```

### Installation via git clone

1. Clone the repository:

   ```bash
   git clone https://github.com/DocsaidLab/Capybara.git
   ```

2. Install the wheel package:

   ```bash
   pip install wheel
   ```

3. Build the wheel file:

   ```bash
   cd Capybara
   python setup.py bdist_wheel
   ```

4. Install the built wheel package:

   ```bash
   pip install dist/capybara-*-py3-none-any.whl
   ```

## Development

For developers who want to contribute to the project, follow these instructions to set up the development environment:

```bash
pip install wheel
pip install -e .
```

### Testing

To ensure the stability and accuracy of Capybara, we use `pytest` for unit testing.

```bash
pip install pytest
```

Users can run the tests themselves to verify the accuracy of the functionalities they are using.

To run the tests:

```bash
python -m pytest -vv tests
```
