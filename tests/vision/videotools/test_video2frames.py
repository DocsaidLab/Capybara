import numpy as np
import pytest

from capybara import get_curdir, video2frames

# 測試用的影片
video_path = get_curdir(__file__) / "video_test.mp4"


def test_video2frames():
    # 測試從影片中提取所有幀
    frames = video2frames(video_path)
    assert isinstance(frames, list)
    assert len(frames) > 0
    assert isinstance(frames[0], np.ndarray)


def test_video2frames_with_fps():
    # 測試指定提取幀的速度
    frames = video2frames(video_path, frame_per_sec=2)
    assert len(frames) == 18


def test_video2frames_with_fps_greater_than_video_fps():
    # When requested FPS exceeds the video FPS, fall back to extracting all frames.
    frames_all = video2frames(video_path)
    frames = video2frames(video_path, frame_per_sec=60)
    assert len(frames) == len(frames_all)


def test_video2frames_rejects_non_positive_fps():
    with pytest.raises(ValueError, match="frame_per_sec must be > 0"):
        video2frames(video_path, frame_per_sec=0)


def test_video2frames_invalid_input():
    # 測試不支援的影片類型
    with pytest.raises(TypeError):
        video2frames("invalid_video.txt")

    # 測試不存在的影片路徑
    with pytest.raises(TypeError):
        video2frames("non_existent_video.mp4")


def test_video2frames_returns_empty_list_when_capture_cannot_open(tmp_path):
    # A file with a valid suffix but invalid contents should fail to open.
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"")
    assert video2frames(bad) == []


def test_video2frames_falls_back_to_all_frames_when_fps_is_invalid(
    monkeypatch, tmp_path
):
    import importlib

    v_mod = importlib.import_module("capybara.vision.videotools.video2frames")

    dummy = tmp_path / "x.mp4"
    dummy.write_bytes(b"0")

    class _FakeCapture:
        def __init__(self) -> None:
            self._idx = 0

        def isOpened(self) -> bool:  # noqa: N802
            return True

        def get(self, prop):
            if prop == v_mod.cv2.CAP_PROP_FPS:
                return 0
            return 0

        def read(self):
            if self._idx >= 3:
                return False, None
            self._idx += 1
            return True, np.zeros((2, 2, 3), dtype=np.uint8)

        def release(self) -> None:
            return None

    monkeypatch.setattr(
        v_mod.cv2, "VideoCapture", lambda *_args, **_kwargs: _FakeCapture()
    )

    frames = v_mod.video2frames(dummy, frame_per_sec=2)
    assert len(frames) == 3
