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
    frames = video2frames_v2(
        video_path, frame_per_sec=2, start_sec=0, end_sec=2, n_threads=2
    )
    assert len(frames) == 4


def test_video2frames_v2_invalid_input():
    # 測試不支援的影片類型
    with pytest.raises(TypeError):
        video2frames_v2("invalid_video.txt")

    # 測試不存在的影片路徑
    with pytest.raises(TypeError):
        video2frames_v2("non_existent_video.mp4")


def test_video2frames_v2_helpers_and_internal_branches(monkeypatch, tmp_path):
    import importlib

    v2_mod = importlib.import_module(
        "capybara.vision.videotools.video2frames_v2"
    )

    assert v2_mod.is_numpy_img(np.zeros((10, 10), dtype=np.uint8))
    assert v2_mod.flatten_list([[1], [2, 3]]) == [1, 2, 3]
    assert v2_mod.flatten_list([[[1], [2]]]) == [1, 2]

    with pytest.raises(ValueError, match="larger than"):
        v2_mod.get_step_inds(0, 1, 5)

    with pytest.raises(TypeError, match="inappropriate"):
        v2_mod._extract_frames([0, 1], "missing.mp4")

    dummy_video = tmp_path / "x.mp4"
    dummy_video.write_bytes(b"")

    resized_calls: list[dict[str, object]] = []
    cvt_calls: list[str] = []

    def fake_imresize(frame: np.ndarray, dsize, **kwargs):
        resized_calls.append({"dsize": dsize, **kwargs})
        h, w = dsize
        return np.zeros((h, w, frame.shape[-1]), dtype=frame.dtype)

    def fake_imcvtcolor(frame: np.ndarray, *, cvt_mode: str) -> np.ndarray:
        cvt_calls.append(cvt_mode)
        return frame

    monkeypatch.setattr(v2_mod, "imresize", fake_imresize)
    monkeypatch.setattr(v2_mod, "imcvtcolor", fake_imcvtcolor)

    class _FakeCapture:
        def __init__(self, frames) -> None:
            self._frames = list(frames)
            self._idx = 0

        def set(self, *_args, **_kwargs) -> None:
            return None

        def read(self):
            if self._idx >= len(self._frames):
                return False, None
            frame = self._frames[self._idx]
            self._idx += 1
            return True, frame

        def release(self) -> None:
            return None

    # scale < 1 branch + color conversion + skip None frame
    monkeypatch.setattr(
        v2_mod.cv2,
        "VideoCapture",
        lambda *_args, **_kwargs: _FakeCapture(
            [None, np.zeros((20, 20, 3), dtype=np.uint8)]
        ),
    )
    frames, _ = v2_mod._extract_frames(
        inds=[0, 1],
        video_path=dummy_video,
        max_size=10,
        color_base="RGB",
    )
    assert len(frames) == 1
    assert resized_calls
    assert cvt_calls == ["BGR2RGB"]

    resized_calls.clear()
    cvt_calls.clear()

    # scale > 1 branch
    monkeypatch.setattr(
        v2_mod.cv2,
        "VideoCapture",
        lambda *_args, **_kwargs: _FakeCapture(
            [np.zeros((5, 5, 3), dtype=np.uint8)]
        ),
    )
    frames2, _ = v2_mod._extract_frames(
        inds=[0],
        video_path=dummy_video,
        max_size=10,
        color_base="BGR",
    )
    assert len(frames2) == 1
    assert any(call.get("interpolation") is not None for call in resized_calls)


def test_video2frames_v2_returns_empty_for_zero_frames_or_fps(
    monkeypatch, tmp_path
):
    import importlib

    v2_mod = importlib.import_module(
        "capybara.vision.videotools.video2frames_v2"
    )

    dummy_video = tmp_path / "x.mp4"
    dummy_video.write_bytes(b"")

    class _FakeCapture:
        def get(self, *_args, **_kwargs):
            return 0

        def release(self) -> None:
            return None

    monkeypatch.setattr(
        v2_mod.cv2, "VideoCapture", lambda *_args, **_kwargs: _FakeCapture()
    )
    assert video2frames_v2(dummy_video) == []


def test_video2frames_v2_validates_start_end_and_handles_worker_exceptions(
    monkeypatch, tmp_path
):
    import importlib

    v2_mod = importlib.import_module(
        "capybara.vision.videotools.video2frames_v2"
    )

    dummy_video = tmp_path / "x.mp4"
    dummy_video.write_bytes(b"")

    class _FakeCapture:
        def __init__(self) -> None:
            self.calls = 0

        def get(self, prop):
            if prop == v2_mod.cv2.CAP_PROP_FRAME_COUNT:
                return 10
            if prop == v2_mod.cv2.CAP_PROP_FPS:
                return 10
            return 0

        def release(self) -> None:
            return None

    monkeypatch.setattr(
        v2_mod.cv2, "VideoCapture", lambda *_args, **_kwargs: _FakeCapture()
    )

    with pytest.raises(ValueError, match="start_sec"):
        video2frames_v2(dummy_video, start_sec=2.0, end_sec=1.0)

    monkeypatch.setattr(
        v2_mod,
        "_extract_frames",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert (
        video2frames_v2(dummy_video, start_sec=0.0, end_sec=1.0, n_threads=1)
        == []
    )


def test_video2frames_v2_skips_empty_worker_chunks(monkeypatch, tmp_path):
    import importlib

    v2_mod = importlib.import_module(
        "capybara.vision.videotools.video2frames_v2"
    )

    dummy_video = tmp_path / "x.mp4"
    dummy_video.write_bytes(b"")

    class _FakeCapture:
        def get(self, prop):
            if prop == v2_mod.cv2.CAP_PROP_FRAME_COUNT:
                return 10
            if prop == v2_mod.cv2.CAP_PROP_FPS:
                return 10
            return 0

        def release(self) -> None:
            return None

    monkeypatch.setattr(
        v2_mod.cv2, "VideoCapture", lambda *_args, **_kwargs: _FakeCapture()
    )

    calls: list[list[int]] = []

    def fake_extract_frames(
        inds,
        video_path,
        max_size=1920,
        color_base="BGR",
        global_ind=0,
    ):
        assert inds, "Should never schedule empty index chunks."
        calls.append(list(inds))
        return [], global_ind

    monkeypatch.setattr(v2_mod, "_extract_frames", fake_extract_frames)

    assert (
        v2_mod.video2frames_v2(
            dummy_video,
            frame_per_sec=10,
            start_sec=0.0,
            end_sec=0.1,
            n_threads=8,
        )
        == []
    )
    assert calls == [[0]]


def test_video2frames_v2_validates_threads_fps_and_zero_duration(
    monkeypatch, tmp_path
):
    import importlib

    v2_mod = importlib.import_module(
        "capybara.vision.videotools.video2frames_v2"
    )

    dummy_video = tmp_path / "x.mp4"
    dummy_video.write_bytes(b"")

    class _FakeCapture:
        def get(self, prop):
            if prop == v2_mod.cv2.CAP_PROP_FRAME_COUNT:
                return 10
            if prop == v2_mod.cv2.CAP_PROP_FPS:
                return 10
            return 0

        def release(self) -> None:
            return None

    monkeypatch.setattr(
        v2_mod.cv2, "VideoCapture", lambda *_args, **_kwargs: _FakeCapture()
    )

    with pytest.raises(ValueError, match="n_threads must be >= 1"):
        v2_mod.video2frames_v2(dummy_video, n_threads=0)

    with pytest.raises(ValueError, match="frame_per_sec must be > 0"):
        v2_mod.video2frames_v2(dummy_video, frame_per_sec=0, n_threads=1)

    assert (
        v2_mod.video2frames_v2(
            dummy_video,
            frame_per_sec=10,
            start_sec=1.0,
            end_sec=1.0,
            n_threads=1,
        )
        == []
    )
