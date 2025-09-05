import time
import datetime
from time import struct_time
import pytest
import numpy as np

from capybara.utils.time import (
    Timer, now, timestamp2datetime, timestamp2time, timestamp2str,
    time2datetime, time2timestamp, time2str, datetime2time,
    datetime2timestamp, datetime2str, str2time, str2datetime, str2timestamp
)


class TestTimer:
    """Test Timer class functionality."""
    
    def test_timer_basic(self):
        """Test basic Timer functionality."""
        timer = Timer()
        assert isinstance(timer, Timer)
        
        # Timer should have precision and other attributes
        assert hasattr(timer, 'precision')
        assert hasattr(timer, 'desc')
        assert hasattr(timer, 'verbose')
        assert hasattr(timer, 'tic')
        assert hasattr(timer, 'toc')
    
    def test_timer_as_context_manager(self):
        """Test Timer as context manager."""
        with Timer() as timer:
            time.sleep(0.01)  # Small delay
        
        # Context manager should return None but still work
        # Timer output goes to stdout, not returned value
        # Just check that it doesn't raise errors
        assert timer is None
    
    def test_timer_manual_timing(self):
        """Test manual timing with Timer."""
        timer = Timer()
        
        # Start the timer
        timer.tic()
        initial_time = timer.time
        
        time.sleep(0.01)  # Small delay
        
        elapsed = timer.toc()
        assert elapsed > 0
        assert elapsed < 1
        
        # time should not change after accessing toc
        assert timer.time == initial_time
    
    def test_timer_multiple_measurements(self):
        """Test multiple measurements with Timer."""
        timer = Timer()
        
        timer.tic()
        time.sleep(0.01)
        elapsed1 = timer.toc()
        
        # Can restart timer for new measurement
        timer.tic()
        time.sleep(0.01)
        elapsed2 = timer.toc()
        
        # Both measurements should be positive
        assert elapsed1 > 0
        assert elapsed2 > 0
    
    def test_timer_str_representation(self):
        """Test Timer string representation."""
        timer = Timer()
        timer.tic()
        time.sleep(0.01)
        timer.toc()
        
        timer_str = str(timer)
        assert isinstance(timer_str, str)
    
    def test_timer_error_handling(self):
        """Test Timer error handling."""
        timer = Timer()
        
        # Should raise error if toc called before tic
        with pytest.raises(ValueError):
            timer.toc()
    
    def test_timer_record_keeping(self):
        """Test Timer record keeping functionality."""
        timer = Timer()
        
        # Make a few measurements
        for _ in range(3):
            timer.tic()
            time.sleep(0.001)
            timer.toc()
        
        # Check that records are kept (Timer has statistical methods)
        assert hasattr(timer, 'mean')
        assert hasattr(timer, 'std')
        assert hasattr(timer, 'min')
        assert hasattr(timer, 'max')
        
        # These should return values, not be methods
        mean_val = timer.mean
        std_val = timer.std
        min_val = timer.min
        max_val = timer.max
        
        # All should be numbers
        assert isinstance(mean_val, (int, float))
        assert isinstance(std_val, (int, float))
        assert isinstance(min_val, (int, float))
        assert isinstance(max_val, (int, float))


class TestNowFunction:
    """Test now() function."""
    
    def test_now_basic(self):
        """Test basic now() functionality."""
        current_time = now()
        assert isinstance(current_time, float)
        assert current_time > 0
    
    def test_now_close_to_time(self):
        """Test that now() is close to time.time()."""
        t1 = now()
        t2 = time.time()
        
        # Should be very close (within 1 second)
        assert abs(t1 - t2) < 1
    
    def test_now_monotonic(self):
        """Test that now() is monotonically increasing."""
        times = [now() for _ in range(5)]
        
        # Each time should be >= previous time
        for i in range(1, len(times)):
            assert times[i] >= times[i-1]


class TestTimestampConversions:
    """Test timestamp conversion functions."""
    
    @pytest.fixture
    def test_timestamp(self):
        """Provide a test timestamp."""
        return 1640995200.0  # 2022-01-01 00:00:00 UTC
    
    def test_timestamp2datetime(self, test_timestamp):
        """Test timestamp to datetime conversion."""
        dt = timestamp2datetime(test_timestamp)
        
        assert isinstance(dt, datetime.datetime)
        assert dt.year == 2022
        assert dt.month == 1
        assert dt.day == 1
    
    def test_timestamp2time(self, test_timestamp):
        """Test timestamp to struct_time conversion."""
        t = timestamp2time(test_timestamp)
        
        assert isinstance(t, struct_time)
        assert t.tm_year == 2022
        assert t.tm_mon == 1
        assert t.tm_mday == 1
    
    def test_timestamp2str(self, test_timestamp):
        """Test timestamp to string conversion."""
        # Default format - need to provide fmt parameter
        s = timestamp2str(test_timestamp, "%Y-%m-%d %H:%M:%S")
        assert isinstance(s, str)
        assert "2022" in s
        
        # Custom format
        s_custom = timestamp2str(test_timestamp, "%Y-%m-%d")
        assert s_custom == "2022-01-01"
    
    def test_timestamp2str_different_formats(self, test_timestamp):
        """Test timestamp to string with different formats."""
        formats = [
            "%Y-%m-%d",
            "%Y-%m-%d %H:%M:%S",
            "%d/%m/%Y",
            "%B %d, %Y"
        ]
        
        for fmt in formats:
            s = timestamp2str(test_timestamp, fmt)
            assert isinstance(s, str)
            assert len(s) > 0


class TestTimeConversions:
    """Test struct_time conversion functions."""
    
    @pytest.fixture
    def test_time(self):
        """Provide a test struct_time."""
        return time.struct_time((2022, 1, 1, 0, 0, 0, 5, 1, 0))
    
    def test_time2datetime(self, test_time):
        """Test struct_time to datetime conversion."""
        dt = time2datetime(test_time)
        
        assert isinstance(dt, datetime.datetime)
        assert dt.year == 2022
        assert dt.month == 1
        assert dt.day == 1
    
    def test_time2timestamp(self, test_time):
        """Test struct_time to timestamp conversion."""
        ts = time2timestamp(test_time)
        
        assert isinstance(ts, float)
        assert ts > 0
    
    def test_time2str(self, test_time):
        """Test struct_time to string conversion."""
        s = time2str(test_time, "%Y-%m-%d %H:%M:%S")
        assert isinstance(s, str)
        assert "2022" in s
        
        # Custom format
        s_custom = time2str(test_time, "%Y-%m-%d")
        assert s_custom == "2022-01-01"


class TestDatetimeConversions:
    """Test datetime conversion functions."""
    
    @pytest.fixture
    def test_datetime(self):
        """Provide a test datetime."""
        return datetime.datetime(2022, 1, 1, 12, 30, 45)
    
    def test_datetime2time(self, test_datetime):
        """Test datetime to struct_time conversion."""
        t = datetime2time(test_datetime)
        
        assert isinstance(t, struct_time)
        assert t.tm_year == 2022
        assert t.tm_mon == 1
        assert t.tm_mday == 1
        assert t.tm_hour == 12
        assert t.tm_min == 30
        assert t.tm_sec == 45
    
    def test_datetime2timestamp(self, test_datetime):
        """Test datetime to timestamp conversion."""
        ts = datetime2timestamp(test_datetime)
        
        assert isinstance(ts, float)
        assert ts > 0
    
    def test_datetime2str(self, test_datetime):
        """Test datetime to string conversion."""
        s = datetime2str(test_datetime, "%Y-%m-%d %H:%M:%S")
        assert isinstance(s, str)
        assert "2022" in s
        
        # Custom format
        s_custom = datetime2str(test_datetime, "%Y-%m-%d %H:%M")
        assert s_custom == "2022-01-01 12:30"


class TestStringConversions:
    """Test string conversion functions."""
    
    @pytest.fixture
    def test_time_string(self):
        """Provide a test time string."""
        return "2022-01-01 12:30:45"
    
    @pytest.fixture
    def test_format(self):
        """Provide the format for test string."""
        return "%Y-%m-%d %H:%M:%S"
    
    def test_str2time(self, test_time_string, test_format):
        """Test string to struct_time conversion."""
        t = str2time(test_time_string, test_format)
        
        assert isinstance(t, struct_time)
        assert t.tm_year == 2022
        assert t.tm_mon == 1
        assert t.tm_mday == 1
        assert t.tm_hour == 12
        assert t.tm_min == 30
        assert t.tm_sec == 45
    
    def test_str2datetime(self, test_time_string, test_format):
        """Test string to datetime conversion."""
        dt = str2datetime(test_time_string, test_format)
        
        assert isinstance(dt, datetime.datetime)
        assert dt.year == 2022
        assert dt.month == 1
        assert dt.day == 1
        assert dt.hour == 12
        assert dt.minute == 30
        assert dt.second == 45
    
    def test_str2timestamp(self, test_time_string, test_format):
        """Test string to timestamp conversion."""
        ts = str2timestamp(test_time_string, test_format)
        
        assert isinstance(ts, float)
        assert ts > 0
    
    def test_str_conversions_different_formats(self):
        """Test string conversions with different formats."""
        test_cases = [
            ("2022-01-01", "%Y-%m-%d"),
            ("01/01/2022", "%m/%d/%Y"),
            ("January 1, 2022", "%B %d, %Y"),
            ("2022-01-01 15:30", "%Y-%m-%d %H:%M")
        ]
        
        for time_str, fmt in test_cases:
            # Should not raise errors
            t = str2time(time_str, fmt)
            dt = str2datetime(time_str, fmt)
            ts = str2timestamp(time_str, fmt)
            
            assert isinstance(t, struct_time)
            assert isinstance(dt, datetime.datetime)
            assert isinstance(ts, float)


class TestRoundTripConversions:
    """Test round-trip conversions to ensure consistency."""
    
    def test_timestamp_roundtrip(self):
        """Test round-trip timestamp conversions."""
        original_ts = time.time()
        
        # timestamp -> datetime -> timestamp
        dt = timestamp2datetime(original_ts)
        roundtrip_ts = datetime2timestamp(dt)
        
        # Should be very close (within 1 second due to precision)
        assert abs(original_ts - roundtrip_ts) < 1
    
    def test_datetime_roundtrip(self):
        """Test round-trip datetime conversions."""
        original_dt = datetime.datetime.now()
        
        # datetime -> timestamp -> datetime
        ts = datetime2timestamp(original_dt)
        roundtrip_dt = timestamp2datetime(ts)
        
        # Should be the same (within 1 second)
        time_diff = abs((original_dt - roundtrip_dt).total_seconds())
        assert time_diff < 1
    
    def test_string_roundtrip(self):
        """Test round-trip string conversions."""
        original_str = "2022-01-01 12:30:45"
        fmt = "%Y-%m-%d %H:%M:%S"
        
        # string -> datetime -> string
        dt = str2datetime(original_str, fmt)
        roundtrip_str = datetime2str(dt, fmt)
        
        assert original_str == roundtrip_str
    
    def test_time_roundtrip(self):
        """Test round-trip struct_time conversions."""
        original_time = time.localtime()
        
        # time -> timestamp -> time
        ts = time2timestamp(original_time)
        roundtrip_time = timestamp2time(ts)
        
        # Should be the same (comparing year, month, day, hour, minute)
        assert original_time.tm_year == roundtrip_time.tm_year
        assert original_time.tm_mon == roundtrip_time.tm_mon
        assert original_time.tm_mday == roundtrip_time.tm_mday
        assert original_time.tm_hour == roundtrip_time.tm_hour
        assert original_time.tm_min == roundtrip_time.tm_min


class TestErrorHandling:
    """Test error handling in time functions."""
    
    def test_invalid_timestamp(self):
        """Test functions with invalid timestamps."""
        # Test with string input
        with pytest.raises((TypeError, ValueError, OSError)):
            timestamp2datetime("not_a_number")
    
    def test_invalid_format_string(self):
        """Test string conversion with invalid format."""
        with pytest.raises(ValueError):
            str2datetime("2022-01-01", "%invalid_format%")
    
    def test_mismatched_string_format(self):
        """Test string conversion with mismatched format."""
        with pytest.raises(ValueError):
            str2datetime("2022-01-01", "%Y-%m-%d %H:%M:%S")  # Missing time part


class TestTimeFunctionIntegration:
    """Test integration between different time functions."""
    
    def test_now_with_conversions(self):
        """Test now() function with various conversions."""
        current_timestamp = now()
        
        # Convert to different formats
        dt = timestamp2datetime(current_timestamp)
        t = timestamp2time(current_timestamp)
        s = timestamp2str(current_timestamp, "%Y-%m-%d %H:%M:%S")
        
        # All should represent the same time
        assert isinstance(dt, datetime.datetime)
        assert isinstance(t, struct_time)
        assert isinstance(s, str)
        
        # Year should be current year (reasonable assumption)
        current_year = datetime.datetime.now().year
        assert dt.year == current_year
        assert t.tm_year == current_year
        assert str(current_year) in s
    
    def test_timer_with_conversions(self):
        """Test Timer with time conversions."""
        timer = Timer()
        timer.tic()
        time.sleep(0.01)
        elapsed = timer.toc()
        
        # Convert elapsed time (which is in seconds) to different formats
        # This is more of a consistency check
        assert elapsed > 0
        assert elapsed < 10  # Should be much less than 10 seconds
        
        # Timer measurements should be reasonable
        assert isinstance(elapsed, float)
    
    def test_all_conversion_functions_exist(self):
        """Test that all advertised conversion functions exist and are callable."""
        functions = [
            timestamp2datetime, timestamp2time, timestamp2str,
            time2datetime, time2timestamp, time2str,
            datetime2time, datetime2timestamp, datetime2str,
            str2time, str2datetime, str2timestamp
        ]
        
        for func in functions:
            assert callable(func), f"Function {func.__name__} is not callable"