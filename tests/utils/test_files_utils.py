import os
import tempfile
import json
import yaml
import pickle
import numpy as np
import pytest
from pathlib import Path

from capybara.utils.files_utils import (
    gen_md5, get_files, load_json, dump_json, load_pickle,
    dump_pickle, load_yaml, dump_yaml, img_to_md5
)


class TestFileUtils:
    """Test class for file utility functions."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def test_file(self, temp_dir):
        """Create a test file with known content."""
        file_path = os.path.join(temp_dir, "test.txt")
        with open(file_path, 'w') as f:
            f.write("Hello, World!")
        return file_path
    
    @pytest.fixture
    def test_image(self):
        """Create a test image array."""
        return np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)


class TestMD5Functions(TestFileUtils):
    """Test MD5 generation functions."""
    
    def test_gen_md5_basic(self, test_file):
        """Test basic MD5 generation."""
        md5_hash = gen_md5(test_file)
        
        assert isinstance(md5_hash, str)
        assert len(md5_hash) == 32  # MD5 hash is 32 characters
        assert all(c in '0123456789abcdef' for c in md5_hash)
    
    def test_gen_md5_consistent(self, test_file):
        """Test that MD5 generation is consistent."""
        md5_1 = gen_md5(test_file)
        md5_2 = gen_md5(test_file)
        
        assert md5_1 == md5_2
    
    def test_gen_md5_different_files(self, temp_dir):
        """Test that different files have different MD5 hashes."""
        file1 = os.path.join(temp_dir, "file1.txt")
        file2 = os.path.join(temp_dir, "file2.txt")
        
        with open(file1, 'w') as f:
            f.write("Content 1")
        with open(file2, 'w') as f:
            f.write("Content 2")
        
        md5_1 = gen_md5(file1)
        md5_2 = gen_md5(file2)
        
        assert md5_1 != md5_2
    
    def test_gen_md5_custom_block_size(self, test_file):
        """Test MD5 generation with custom block size."""
        md5_default = gen_md5(test_file)
        md5_custom = gen_md5(test_file, block_size=64)
        
        # Should produce the same result regardless of block size
        assert md5_default == md5_custom
    
    def test_img_to_md5_basic(self, test_image):
        """Test MD5 generation for images."""
        md5_hash = img_to_md5(test_image)
        
        assert isinstance(md5_hash, str)
        assert len(md5_hash) == 32
        assert all(c in '0123456789abcdef' for c in md5_hash)
    
    def test_img_to_md5_consistent(self, test_image):
        """Test that image MD5 generation is consistent."""
        md5_1 = img_to_md5(test_image)
        md5_2 = img_to_md5(test_image)
        
        assert md5_1 == md5_2
    
    def test_img_to_md5_different_images(self):
        """Test that different images have different MD5 hashes."""
        img1 = np.zeros((10, 10), dtype=np.uint8)
        img2 = np.ones((10, 10), dtype=np.uint8)
        
        md5_1 = img_to_md5(img1)
        md5_2 = img_to_md5(img2)
        
        assert md5_1 != md5_2
    
    def test_img_to_md5_invalid_input(self):
        """Test img_to_md5 with invalid input."""
        with pytest.raises(TypeError):
            img_to_md5("not an array")
        
        with pytest.raises(TypeError):
            img_to_md5([1, 2, 3])


class TestGetFiles(TestFileUtils):
    """Test get_files function."""
    
    def test_get_files_basic(self, temp_dir):
        """Test basic file listing."""
        # Create some test files
        files = ["test1.txt", "test2.py", "test3.jpg"]
        for file in files:
            with open(os.path.join(temp_dir, file), 'w') as f:
                f.write("test")
        
        result = get_files(temp_dir)
        
        assert isinstance(result, list)
        assert len(result) == 3
        
        # Check that all files are found
        result_names = [os.path.basename(f) for f in result]
        for file in files:
            assert file in result_names
    
    def test_get_files_with_suffix(self, temp_dir):
        """Test file listing with suffix filter."""
        # Create files with different extensions
        files = ["test1.txt", "test2.py", "test3.txt", "test4.jpg"]
        for file in files:
            with open(os.path.join(temp_dir, file), 'w') as f:
                f.write("test")
        
        result = get_files(temp_dir, suffix=".txt")
        
        assert len(result) == 2
        # Convert to strings for suffix checking since get_files returns Path objects
        result_strs = [str(f) for f in result]
        assert all(f.endswith('.txt') for f in result_strs)
    
    def test_get_files_recursive(self, temp_dir):
        """Test recursive file listing."""
        # Create subdirectory with files
        subdir = os.path.join(temp_dir, "subdir")
        os.makedirs(subdir)
        
        with open(os.path.join(temp_dir, "file1.txt"), 'w') as f:
            f.write("test")
        with open(os.path.join(subdir, "file2.txt"), 'w') as f:
            f.write("test")
        
        result = get_files(temp_dir, suffix=".txt")
        
        assert len(result) == 2
        # Convert to strings for checking
        result_strs = [str(f) for f in result]
        assert any("file1.txt" in f for f in result_strs)
        assert any("file2.txt" in f for f in result_strs)
    
    def test_get_files_empty_directory(self, temp_dir):
        """Test get_files on empty directory."""
        result = get_files(temp_dir)
        
        assert isinstance(result, list)
        assert len(result) == 0


class TestJSONFunctions(TestFileUtils):
    """Test JSON load/dump functions."""
    
    def test_dump_load_json_basic(self, temp_dir):
        """Test basic JSON dump and load."""
        data = {"key": "value", "number": 42, "list": [1, 2, 3]}
        file_path = os.path.join(temp_dir, "test.json")
        
        # Dump data
        dump_json(data, file_path)
        
        # Check file exists
        assert os.path.exists(file_path)
        
        # Load data
        loaded_data = load_json(file_path)
        
        assert loaded_data == data
    
    def test_json_complex_data(self, temp_dir):
        """Test JSON with complex data structures."""
        data = {
            "nested": {"key": "value"},
            "list": [1, 2, {"inner": "data"}],
            "null": None,
            "bool": True,
            "float": 3.14
        }
        file_path = os.path.join(temp_dir, "complex.json")
        
        dump_json(data, file_path)
        loaded_data = load_json(file_path)
        
        assert loaded_data == data
    
    def test_json_with_kwargs(self, temp_dir):
        """Test JSON functions with additional kwargs."""
        data = {"key": "value"}
        file_path = os.path.join(temp_dir, "test.json")
        
        # Dump with indentation
        dump_json(data, file_path, indent=2)
        
        # Load data
        loaded_data = load_json(file_path)
        
        assert loaded_data == data
        
        # Check that file is properly formatted
        with open(file_path, 'r') as f:
            content = f.read()
            assert "  " in content  # Should have indentation
    
    def test_load_json_nonexistent_file(self, temp_dir):
        """Test loading non-existent JSON file."""
        file_path = os.path.join(temp_dir, "nonexistent.json")
        
        with pytest.raises(FileNotFoundError):
            load_json(file_path)


class TestPickleFunctions(TestFileUtils):
    """Test Pickle load/dump functions."""
    
    def test_dump_load_pickle_basic(self, temp_dir):
        """Test basic pickle dump and load."""
        data = {"key": "value", "number": 42, "array": np.array([1, 2, 3])}
        file_path = os.path.join(temp_dir, "test.pkl")
        
        # Dump data
        dump_pickle(data, file_path)
        
        # Check file exists
        assert os.path.exists(file_path)
        
        # Load data
        loaded_data = load_pickle(file_path)
        
        assert loaded_data["key"] == data["key"]
        assert loaded_data["number"] == data["number"]
        np.testing.assert_array_equal(loaded_data["array"], data["array"])
    
    def test_pickle_numpy_array(self, temp_dir, test_image):
        """Test pickling numpy arrays."""
        file_path = os.path.join(temp_dir, "array.pkl")
        
        dump_pickle(test_image, file_path)
        loaded_array = load_pickle(file_path)
        
        np.testing.assert_array_equal(loaded_array, test_image)
    
    def test_pickle_complex_objects(self, temp_dir):
        """Test pickling complex Python objects."""
        # Use simpler test that doesn't rely on class equality
        data = {
            "number": 42,
            "function": lambda x: x * 2,
            "set": {1, 2, 3},
            "list": [1, 2, 3]
        }
        file_path = os.path.join(temp_dir, "complex.pkl")
        
        dump_pickle(data, file_path)
        loaded_data = load_pickle(file_path)
        
        # Check basic data types
        assert loaded_data["number"] == 42
        assert loaded_data["function"](5) == 10
        assert loaded_data["set"] == {1, 2, 3}
        assert loaded_data["list"] == [1, 2, 3]


class TestYAMLFunctions(TestFileUtils):
    """Test YAML load/dump functions."""
    
    def test_dump_load_yaml_basic(self, temp_dir):
        """Test basic YAML dump and load."""
        data = {"key": "value", "number": 42, "list": [1, 2, 3]}
        file_path = os.path.join(temp_dir, "test.yaml")
        
        # Dump data
        dump_yaml(data, file_path)
        
        # Check file exists
        assert os.path.exists(file_path)
        
        # Load data
        loaded_data = load_yaml(file_path)
        
        assert loaded_data == data
    
    def test_yaml_complex_data(self, temp_dir):
        """Test YAML with complex data structures."""
        data = {
            "nested": {"key": "value"},
            "list": [1, 2, {"inner": "data"}],
            "null": None,
            "bool": True,
            "float": 3.14
        }
        file_path = os.path.join(temp_dir, "complex.yaml")
        
        dump_yaml(data, file_path)
        loaded_data = load_yaml(file_path)
        
        assert loaded_data == data
    
    def test_yaml_with_kwargs(self, temp_dir):
        """Test YAML functions with additional kwargs."""
        data = {"key": "value", "list": [1, 2, 3]}
        file_path = os.path.join(temp_dir, "test.yaml")
        
        # Dump with specific formatting
        dump_yaml(data, file_path, default_flow_style=False)
        
        # Load data
        loaded_data = load_yaml(file_path)
        
        assert loaded_data == data
    
    def test_load_yaml_nonexistent_file(self, temp_dir):
        """Test loading non-existent YAML file."""
        file_path = os.path.join(temp_dir, "nonexistent.yaml")
        
        with pytest.raises(FileNotFoundError):
            load_yaml(file_path)


class TestPathHandling(TestFileUtils):
    """Test path handling in file functions."""
    
    def test_functions_with_path_objects(self, temp_dir):
        """Test that functions work with Path objects."""
        data = {"test": "data"}
        file_path = Path(temp_dir) / "test.json"
        
        # Should work with Path objects
        dump_json(data, file_path)
        loaded_data = load_json(file_path)
        
        assert loaded_data == data
    
    def test_functions_with_string_paths(self, temp_dir):
        """Test that functions work with string paths."""
        data = {"test": "data"}
        file_path = os.path.join(temp_dir, "test.json")
        
        # Should work with string paths
        dump_json(data, file_path)
        loaded_data = load_json(file_path)
        
        assert loaded_data == data


class TestErrorHandling(TestFileUtils):
    """Test error handling in file functions."""
    
    def test_gen_md5_nonexistent_file(self):
        """Test gen_md5 with non-existent file."""
        with pytest.raises(FileNotFoundError):
            gen_md5("nonexistent_file.txt")
    
    def test_get_files_nonexistent_directory(self):
        """Test get_files with non-existent directory."""
        with pytest.raises(FileNotFoundError):
            get_files("nonexistent_directory")
    
    def test_dump_json_invalid_data(self, temp_dir):
        """Test dumping non-serializable data to JSON."""
        file_path = os.path.join(temp_dir, "test.json")
        
        # Functions/lambdas are not JSON serializable
        with pytest.raises(TypeError):
            dump_json({"func": lambda x: x}, file_path)
    
    def test_load_invalid_json(self, temp_dir):
        """Test loading invalid JSON."""
        file_path = os.path.join(temp_dir, "invalid.json")
        
        # Write invalid JSON
        with open(file_path, 'w') as f:
            f.write("invalid json content {")
        
        # The actual library used might be ujson, which has a different exception
        with pytest.raises((json.JSONDecodeError, ValueError)):
            load_json(file_path)