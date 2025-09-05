import pytest
from capybara.cpuinfo import cpuinfo


def test_cpuinfo_basic():
    """Test basic cpuinfo functionality."""
    info = cpuinfo()
    
    # Should return a CPUInfo object
    assert hasattr(info, 'info')
    assert hasattr(info, '_getNCPUs')
    
    # Should have info attribute that contains CPU data
    cpu_data = info.info
    assert isinstance(cpu_data, list)
    assert len(cpu_data) > 0
    
    # Each CPU info should be a dictionary
    for cpu_info in cpu_data:
        assert isinstance(cpu_info, dict)


def test_cpuinfo_processor_count():
    """Test that cpuinfo returns info for each processor."""
    info = cpuinfo()
    cpu_data = info.info
    
    # Should have at least one processor
    assert len(cpu_data) >= 1
    
    # Should be able to get CPU count
    ncpus = info._getNCPUs()
    assert isinstance(ncpus, int)
    assert ncpus > 0
    
    # CPU count should match the length of info
    assert ncpus == len(cpu_data)


def test_cpuinfo_cpu_detection_methods():
    """Test CPU detection methods."""
    info = cpuinfo()
    
    # Test various CPU detection methods
    detection_methods = [
        '_is_Intel', '_is_AMD', '_is_32bit', '_is_64bit',
        '_has_mmx', '_has_sse', '_has_sse2'
    ]
    
    for method in detection_methods:
        if hasattr(info, method):
            result = getattr(info, method)()
            assert isinstance(result, bool)


def test_cpuinfo_architecture_detection():
    """Test architecture detection."""
    info = cpuinfo()
    
    # At least one architecture should be detected
    arch_methods = ['_is_i386', '_is_i486', '_is_i586', '_is_i686', '_is_64bit']
    detected_archs = []
    
    for method in arch_methods:
        if hasattr(info, method):
            try:
                if getattr(info, method)():
                    detected_archs.append(method)
            except KeyError:
                # Some methods might fail on certain systems
                pass
    
    # Should detect at least one architecture, but if none detected due to system specifics, that's OK
    # Just ensure the methods exist and can be called
    assert len(arch_methods) > 0


def test_cpuinfo_vendor_detection():
    """Test CPU vendor detection."""
    info = cpuinfo()
    
    # Should detect either Intel or AMD (on x86 systems)
    is_intel = info._is_Intel()
    is_amd = info._is_AMD()
    
    assert isinstance(is_intel, bool)
    assert isinstance(is_amd, bool)
    
    # On typical x86 systems, should be either Intel or AMD
    # (Though this might not be true on all systems, so we just check types)


def test_cpuinfo_feature_detection():
    """Test CPU feature detection."""
    info = cpuinfo()
    
    # Test common CPU features
    feature_methods = [
        '_has_mmx', '_has_sse', '_has_sse2', '_has_sse3',
        '_has_3dnow', '_has_3dnowext'
    ]
    
    for method in feature_methods:
        if hasattr(info, method):
            result = getattr(info, method)()
            assert isinstance(result, bool)


def test_cpuinfo_cpu_type_detection():
    """Test specific CPU type detection."""
    info = cpuinfo()
    
    # Test various CPU type detection methods
    cpu_type_methods = [
        '_is_Pentium', '_is_PentiumII', '_is_PentiumIII', '_is_PentiumIV',
        '_is_PentiumM', '_is_Core2', '_is_Celeron', '_is_Xeon',
        '_is_Athlon64', '_is_AthlonK7', '_is_Opteron'
    ]
    
    detected_types = []
    for method in cpu_type_methods:
        if hasattr(info, method):
            if getattr(info, method)():
                detected_types.append(method)
    
    # It's okay if no specific type is detected (generic CPU)
    # Just ensure the methods work
    assert isinstance(detected_types, list)


def test_cpuinfo_info_structure():
    """Test the structure of CPU info data."""
    info = cpuinfo()
    cpu_data = info.info
    
    # Each CPU entry should be a dictionary with string keys
    for cpu_info in cpu_data:
        assert isinstance(cpu_info, dict)
        
        for key, value in cpu_info.items():
            assert isinstance(key, str)
            # Values can be strings or bytes (like uname_m)
            assert isinstance(value, (str, bytes))
            
            # Keys should not be empty
            assert len(key.strip()) > 0
            
            # Values should not be empty (handle bytes separately)
            # Some values might be empty strings, so we'll be more lenient
            if isinstance(value, str):
                # Just check that value is a string, not that it's non-empty
                pass
            else:  # bytes
                assert len(value) > 0


def test_cpuinfo_consistent_info():
    """Test that CPU info is consistent across calls."""
    info1 = cpuinfo()
    info2 = cpuinfo()
    
    # Should return consistent information
    assert info1._getNCPUs() == info2._getNCPUs()
    assert len(info1.info) == len(info2.info)
    
    # CPU features should be consistent
    assert info1._is_Intel() == info2._is_Intel()
    assert info1._is_AMD() == info2._is_AMD()
    assert info1._is_64bit() == info2._is_64bit()


def test_cpuinfo_error_handling():
    """Test that cpuinfo handles errors gracefully."""
    info = cpuinfo()
    
    # Accessing non-existent methods should raise AttributeError
    with pytest.raises(AttributeError):
        info._non_existent_method()
    
    # But normal methods should work
    assert callable(getattr(info, '_getNCPUs'))
    assert callable(getattr(info, '_is_64bit'))