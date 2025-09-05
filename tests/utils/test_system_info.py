import shutil
from unittest.mock import MagicMock, patch

import pytest
import requests

from capybara.utils.system_info import (
    get_cpu_info,
    # get_external_ip,
    # get_gpu_cuda_versions,
    get_gpu_lib_info,
    get_package_versions,
    get_system_info,
)


def test_get_package_versions_basic():
    """Test basic package version retrieval."""
    versions = get_package_versions()

    assert isinstance(versions, dict)
    assert len(versions) > 0

    # Should have entries for common packages (even if they're errors)
    expected_packages = ["PyTorch", "TensorFlow", "Keras", "NumPy", "OpenCV"]

    for package in expected_packages:
        # Should have either a version or an error for each package
        version_key = f"{package} Version"
        error_key = f"{package} Error"
        assert version_key in versions or error_key in versions, f"No information found for {package}"


def test_get_package_versions_numpy():
    """Test that numpy version is always available (since it's a dependency)."""
    versions = get_package_versions()

    # NumPy should always be available since it's a dependency
    assert "NumPy Version" in versions
    assert isinstance(versions["NumPy Version"], str)
    assert len(versions["NumPy Version"]) > 0


def test_get_package_versions_opencv():
    """Test that OpenCV version is available (since it's a dependency)."""
    versions = get_package_versions()

    # OpenCV should be available since it's a dependency
    assert "OpenCV Version" in versions
    assert isinstance(versions["OpenCV Version"], str)
    assert len(versions["OpenCV Version"]) > 0


def test_get_gpu_lib_info():
    """Test GPU CUDA version detection when nvidia-smi fails."""

    versions = get_gpu_lib_info()

    assert isinstance(versions, dict)
    # Should have some CUDA-related information even if it's an error
    assert any("CUDA" in key or "NVIDIA" in key for key in versions.keys())


def test_get_system_info_basic():
    """Test basic system information retrieval."""
    info = get_system_info()

    assert isinstance(info, dict)

    # Should have basic system information based on actual API
    expected_keys = ["OS Version", "CPU Model", "Physical CPU Cores", "Total RAM (GB)", "Disk Total of / (GB)"]
    for key in expected_keys:
        assert key in info, f"Missing system info key: {key}"


def test_get_system_info_detailed():
    """Test detailed system information."""
    info = get_system_info()

    # Check that we have reasonable values
    assert "CPU Model" in info
    assert isinstance(info["CPU Model"], str)
    assert len(info["CPU Model"]) > 0

    assert "Physical CPU Cores" in info
    assert isinstance(info["Physical CPU Cores"], int)
    assert info["Physical CPU Cores"] > 0

    assert "Total RAM (GB)" in info
    assert isinstance(info["Total RAM (GB)"], (int, float))
    assert info["Total RAM (GB)"] > 0


def test_get_cpu_info_basic():
    """Test basic CPU information retrieval."""
    info = get_cpu_info()

    # Based on the actual API, this returns a string
    assert isinstance(info, str)
    assert len(info) > 0

    # Should contain CPU model information
    assert any(term in info.lower() for term in ["cpu", "processor", "intel", "amd", "core"])


# @patch("capybara.utils.system_info.requests.get")
# def test_get_external_ip_success(mock_get):
#     """Test external IP retrieval when successful."""
#     # Mock successful response - just test that it doesn't crash
#     mock_response = MagicMock()
#     mock_response.json.return_value = {"origin": "192.168.1.1"}
#     mock_response.raise_for_status.return_value = None
#     mock_get.return_value = mock_response

#     ip = get_external_ip()

#     # The actual function might return different format, just check it's a string
#     assert isinstance(ip, str)
#     # Should either contain an IP or be an error message
#     assert len(ip) > 0


# @patch("capybara.utils.system_info.requests.get")
# def test_get_external_ip_failure(mock_get):
#     """Test external IP retrieval when it fails."""
#     # Mock failed request
#     mock_get.side_effect = requests.RequestException("Network error")

#     ip = get_external_ip()

#     assert "Error obtaining IP" in ip


# @patch("capybara.utils.system_info.requests.get")
# def test_get_external_ip_timeout(mock_get):
#     """Test external IP retrieval with timeout."""
#     # Mock timeout
#     mock_get.side_effect = requests.Timeout("Request timeout")

#     ip = get_external_ip()

#     assert "Error obtaining IP" in ip


def test_system_integration():
    """Test that all system info functions work together."""
    # Get all system information
    package_versions = get_package_versions()
    gpu_info = get_gpu_lib_info()
    system_info = get_system_info()
    cpu_info = get_cpu_info()

    # Package versions and gpu versions should be dictionaries
    assert isinstance(package_versions, dict)
    assert isinstance(gpu_info, dict)
    # System info should be a dictionary
    assert isinstance(system_info, dict)
    # CPU info should be a string
    assert isinstance(cpu_info, str)

    # All should have some content
    assert len(package_versions) > 0
    assert len(gpu_info) > 0
    assert len(system_info) > 0
    assert len(cpu_info) > 0

    # Combining dict info should work
    combined_dict_info = {
        "packages": package_versions,
        "gpu": gpu_info,
        "system": system_info,
    }

    assert len(combined_dict_info) == 3
    assert all(isinstance(v, dict) for v in combined_dict_info.values())
