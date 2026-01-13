from __future__ import annotations

import itertools

import numpy as np


def test_ipcam_capture_is_an_iterator_and_yields_frames(monkeypatch):
    import capybara.vision.ipcam.camera as cam_mod

    class _FakeCapture:
        def get(self, prop):
            # 4: height, 3: width (as used by the implementation)
            if prop == 4:
                return 10
            if prop == 3:
                return 20
            return 0

        def read(self):
            return False, None

    class _FakeThread:
        def __init__(self, target, daemon=True):
            self._target = target

        def start(self):
            self._target()

    monkeypatch.setattr(
        cam_mod.cv2, "VideoCapture", lambda *_args, **_kwargs: _FakeCapture()
    )
    monkeypatch.setattr(cam_mod, "Thread", _FakeThread)

    cap = cam_mod.IpcamCapture(url="fake", color_base="BGR")
    assert iter(cap) is cap

    frames = list(itertools.islice(cap, 3))
    assert len(frames) == 3
    assert all(isinstance(frame, np.ndarray) for frame in frames)


def test_ipcam_capture_rejects_unsupported_image_size(monkeypatch):
    import capybara.vision.ipcam.camera as cam_mod

    class _FakeCapture:
        def get(self, prop):
            if prop == 4:
                return 0
            if prop == 3:
                return 0
            return 0

        def read(self):
            return False, None

    class _FakeThread:
        def __init__(self, target, daemon=True):
            self._target = target

        def start(self):
            self._target()

    monkeypatch.setattr(
        cam_mod.cv2, "VideoCapture", lambda *_args, **_kwargs: _FakeCapture()
    )
    monkeypatch.setattr(cam_mod, "Thread", _FakeThread)

    with np.testing.assert_raises(ValueError):
        cam_mod.IpcamCapture(url="fake", color_base="BGR")


def test_ipcam_capture_converts_color_and_returns_frame_copy(monkeypatch):
    import capybara.vision.ipcam.camera as cam_mod

    calls: list[str] = []

    def fake_imcvtcolor(frame: np.ndarray, *, cvt_mode: str) -> np.ndarray:
        calls.append(cvt_mode)
        return frame + 1

    class _FakeCapture:
        def __init__(self):
            self._calls = 0

        def get(self, prop):
            if prop == 4:
                return 10
            if prop == 3:
                return 20
            return 0

        def read(self):
            if self._calls == 0:
                self._calls += 1
                return True, np.zeros((10, 20, 3), dtype=np.uint8)
            return False, None

    class _FakeThread:
        def __init__(self, target, daemon=True):
            self._target = target

        def start(self):
            self._target()

    monkeypatch.setattr(
        cam_mod.cv2, "VideoCapture", lambda *_args, **_kwargs: _FakeCapture()
    )
    monkeypatch.setattr(cam_mod, "Thread", _FakeThread)
    monkeypatch.setattr(cam_mod, "imcvtcolor", fake_imcvtcolor)

    cap = cam_mod.IpcamCapture(url="fake", color_base="RGB")
    frame1 = cap.get_frame()
    assert calls == ["BGR2RGB"]
    assert frame1.sum() > 0

    frame1[0, 0, 0] = 123
    frame2 = cap.get_frame()
    assert frame2[0, 0, 0] != 123
