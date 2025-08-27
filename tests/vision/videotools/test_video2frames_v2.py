import numpy as np
import pytest

from capybara import get_curdir, video2frames_v2

# 測試用的影片
video_path = get_curdir(__file__) / "video_test.mp4"


def test_video2frames_v2():
    # 測試從影片中提取所有幀
    frames = video2frames_v2(video_path)
    assert isinstance(frames, list)
    assert len(frames) > 0
    assert isinstance(frames[0], np.ndarray)


def test_video2frames_v2_with_fps():
    # 測試指定提取幀的速度
    frames = video2frames_v2(video_path, frame_per_sec=2, start_sec=0, end_sec=2, n_threads=2)
    assert len(frames) == 4


def test_video2frames_v2_invalid_input():
    # 測試不支援的影片類型
    with pytest.raises(TypeError):
        video2frames_v2("invalid_video.txt")

    # 測試不存在的影片路徑
    with pytest.raises(TypeError):
        video2frames_v2("non_existent_video.mp4")
