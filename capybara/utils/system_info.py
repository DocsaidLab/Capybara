import json
import platform
import socket
import subprocess

import psutil
import requests

__all__ = [
    "get_package_versions",
    "get_gpu_cuda_versions",
    "get_gpu_lib_info",
    "get_system_info",
    "get_cpu_info",
    "get_external_ip",
]


def get_os_version():
    system = platform.system()
    release = platform.release()
    version = platform.version()

    if system == "Linux":
        try:
            import distro

            # Example: "Ubuntu 24.04 LTS (6.8.0-41-generic)"
            return f"{distro.name(pretty=True)} ({release})"
        except ImportError:
            # Fallback if distro not installed
            return f"{system} {release} ({version})"
    elif system == "Darwin":
        # macOS
        mac_ver = platform.mac_ver()[0]
        return f"macOS {mac_ver}"
    elif system == "Windows":
        return f"Windows {release} (Build {version})"
    else:
        return f"{system} {release} ({version})"


def get_package_versions():
    """
    Get versions of commonly used packages in deep learning and data science.

    Returns:
        dict: Dictionary containing versions of installed packages.
    """
    versions_info = {}

    # PyTorch
    try:
        import torch

        versions_info["PyTorch Version"] = torch.__version__
    except Exception as e:
        versions_info["PyTorch Error"] = str(e)

    # PyTorch Lightning
    try:
        import pytorch_lightning as pl

        versions_info["PyTorch Lightning Version"] = pl.__version__
    except Exception as e:
        versions_info["PyTorch Lightning Error"] = str(e)

    # TensorFlow
    try:
        import tensorflow as tf

        versions_info["TensorFlow Version"] = tf.__version__
    except Exception as e:
        versions_info["TensorFlow Error"] = str(e)

    # Keras
    try:
        import keras

        versions_info["Keras Version"] = keras.__version__
    except Exception as e:
        versions_info["Keras Error"] = str(e)

    # NumPy
    try:
        import numpy as np

        versions_info["NumPy Version"] = np.__version__
    except Exception as e:
        versions_info["NumPy Error"] = str(e)

    # Pandas
    try:
        import pandas as pd

        versions_info["Pandas Version"] = pd.__version__
    except Exception as e:
        versions_info["Pandas Error"] = str(e)

    # Scikit-learn
    try:
        import sklearn

        versions_info["Scikit-learn Version"] = sklearn.__version__
    except Exception as e:
        versions_info["Scikit-learn Error"] = str(e)

    # OpenCV
    try:
        import cv2

        versions_info["OpenCV Version"] = cv2.__version__
    except Exception as e:
        versions_info["OpenCV Error"] = str(e)

    # ... and so on for any other packages you're interested in

    return versions_info


def get_gpu_cuda_versions():
    """
    Get GPU and CUDA versions using popular Python libraries.

    Returns:
        dict: Dictionary containing CUDA and GPU driver versions.
    """

    # Attempt to retrieve CUDA version using PyTorch
    try:
        import torch

        torch_cuda_version = torch.version.cuda
    except ImportError:
        torch_cuda_version = "PyTorch not installed"

    # If not retrieved via PyTorch, try using TensorFlow
    try:
        import tensorflow as tf

        tf_cuda_version = tf.version.COMPILER_VERSION
    except ImportError:
        tf_cuda_version = "TensorFlow not installed"

    # If still not retrieved, try using CuPy
    try:
        import cupy

        cupy_cuda_version = cupy.cuda.runtime.runtimeGetVersion()
    except ImportError:
        cupy_cuda_version = "CuPy not installed"

    import onnxruntime as ort

    ort_cuda_version = ort.cuda_version if ort.get_device() == "GPU" else "ONNX Runtime not using GPU"

    # Try to get Nvidia driver version using nvidia-smi command
    try:
        smi_output = subprocess.check_output(["nvidia-smi", "-q"]).decode("utf-8").strip().split("\n")
        nvidia_driver_cuda = [line for line in smi_output if "CUDA Version" in line][0].split(":")[1].strip()
    except Exception as e:
        nvidia_driver_cuda = f"Error getting NVIDIA driver information: {e}"

    return {
        "NVIDIA SMI - CUDA Version": nvidia_driver_cuda,
        "PyTorch CUDA Version": torch_cuda_version,
        "TensorFlow CUDA Version": tf_cuda_version,
        "CuPy CUDA Version": cupy_cuda_version,
        "ONNX Runtime CUDA Version": ort_cuda_version,
    }


def _run(cmd):
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode("utf-8", errors="replace").strip()
    except Exception:
        return None


def get_gpu_lib_info():
    """
    Get GPU info with CUDA version (if NVIDIA) + PyTorch & ONNX Runtime CUDA versions.
    Returns a dict.
    """
    system = platform.system()
    gpus = []
    nvidia_driver_version = None
    nvidia_cuda_version = None

    # -------------------
    # System GPU detection
    # -------------------
    if system == "Darwin":  # macOS
        sp = _run(["system_profiler", "SPDisplaysDataType", "-json"])
        if sp:
            try:
                data = json.loads(sp).get("SPDisplaysDataType", [])
                for d in data:
                    model = d.get("_name")
                    vendor = d.get("spdisplays_vendor")
                    metal = d.get("spdisplays_metal")
                    gpus.append(", ".join([x for x in [model, vendor, f"Metal: {metal}"] if x]))
            except Exception:
                pass

    elif system == "Linux":
        # NVIDIA GPUs via nvidia-smi
        q = _run(["nvidia-smi", "-q"]).split("\n")
        if q:
            lines = [ln.strip() for ln in q if ln.strip()]
            nvidia_driver_version = [ln.split(":")[-1].strip() for ln in lines if "Driver Version" in ln][0]
            nvidia_cuda_version = [ln.split(":")[-1].strip() for ln in lines if "CUDA Version" in ln][0]
            gpus = [ln.split(":")[-1].strip() for ln in lines if "Product Name" in ln]

        # Fallback on Linux for non-NVIDIA GPUs
        if not gpus and system == "Linux":
            pci = _run(["bash", "-lc", "command -v lspci >/dev/null && lspci | egrep 'VGA|3D|Display'"])
            if pci:
                gpus = [ln.split(":")[-1].strip() for ln in pci.splitlines()]

    else:
        raise NotImplementedError(f"Unsupported platform: {system}")

    # -------------------
    # PyTorch CUDA version
    # -------------------
    torch_cuda_version = None
    torch_cudnn_version = None
    try:
        import torch

        torch_cuda_version = torch.version.cuda
        torch_cudnn_version = getattr(torch.backends.cudnn, "version", lambda: None)()
    except Exception:
        pass

    # -------------------
    # ONNX Runtime CUDA provider
    # -------------------
    ort_version = None
    ort_providers = []
    try:
        import onnxruntime as ort

        ort_version = ort.version
        ort_providers = ort.get_available_providers()
    except Exception:
        pass

    return {
        "GPUs": gpus,
        "NVIDIA": {
            "Driver Version": nvidia_driver_version,
            "CUDA Version": nvidia_cuda_version,
        },
        "PyTorch": {
            "CUDA Version": torch_cuda_version,
            "CUDNN Version": torch_cudnn_version,
        },
        "ONNX Runtime": {
            "Version": ort_version,
            "Providers": ort_providers,
            "CUDA Version": ort.cuda_version if ort_providers and "CUDAExecutionProvider" in ort_providers else None,
        },
    }


def get_cpu_info():
    """
    Retrieve the CPU model name based on the platform.

    Returns:
        str: CPU model name or "N/A" if not found.
    """
    if platform.system() == "Windows":
        return platform.processor()
    elif platform.system() == "Darwin":
        # For macOS
        command = "sysctl -n machdep.cpu.brand_string"
        return subprocess.check_output(command, shell=True).strip().decode()
    elif platform.system() == "Linux":
        # For Linux
        command = "cat /proc/cpuinfo | grep 'model name' | uniq"
        return subprocess.check_output(command, shell=True).strip().decode().split(":")[1].strip()
    else:
        return "N/A"


def get_external_ip():
    try:
        response = requests.get("https://httpbin.org/ip")
        return response.json()["origin"]
    except Exception as e:
        return f"Error obtaining IP: {e}"


def get_system_info():
    """
    Fetch system information like OS version, CPU info, RAM, Disk usage, etc.

    Returns:
        dict: Dictionary containing system information.
    """
    info = {
        "OS Version": get_os_version(),
        "CPU Model": get_cpu_info(),
        "Physical CPU Cores": psutil.cpu_count(logical=False),
        "Logical CPU Cores (incl. hyper-threading)": psutil.cpu_count(logical=True),
        "Total RAM (GB)": round(psutil.virtual_memory().total / (1024**3), 2),
        "Available RAM (GB)": round(psutil.virtual_memory().available / (1024**3), 2),
        "Disk Total of / (GB)": round(psutil.disk_usage("/").total / (1024**3), 2),
        "Disk Used of / (GB)": round(psutil.disk_usage("/").used / (1024**3), 2),
        "Disk Free of / (GB)": round(psutil.disk_usage("/").free / (1024**3), 2),
    }

    # Try to fetch GPU information using nvidia-smi command
    try:
        info["GPUs"] = get_gpu_lib_info()["GPUs"]
    except Exception:
        info["GPUs"] = "N/A or Error"

    # Get network information (robust to restricted environments)
    try:
        net = psutil.net_if_addrs()
    except Exception:
        net = {}

    # make enp130s0 workable for some systems
    addrs = net.get("enp130s0", [])
    addrs += net.get("enp5s0", [])

    if len(addrs):
        info["IPV4 Address (Internal)"] = [
            addr.address for addr in addrs if getattr(addr, "family", None) == socket.AF_INET
        ]
    else:
        info["IPV4 Address (Internal)"] = []
    info["IPV4 Address (External)"] = get_external_ip()

    # Determine platform and choose correct address family for MAC
    if hasattr(socket, "AF_LINK"):
        AF_LINK = socket.AF_LINK
    elif hasattr(psutil, "AF_LINK"):
        AF_LINK = psutil.AF_LINK
    else:
        raise Exception("Cannot determine the correct AF_LINK value for this platform.")

    if len(addrs):
        info["MAC Address"] = [addr.address for addr in addrs if getattr(addr, "family", None) == AF_LINK]
    else:
        info["MAC Address"] = []

    return info
