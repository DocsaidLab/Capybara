from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import chain
from typing import Any, List

import cv2
import numpy as np

from ..functionals import imcvtcolor, imresize
from .video2frames import is_video_file

__all__ = ["video2frames_v2"]


def is_numpy_img(x: Any) -> bool:
    """
    x == ndarray (H x W x C)
    """
    return isinstance(x, np.ndarray) and (x.ndim == 2 or (x.ndim == 3 and x.shape[-1] in [1, 3]))


def flatten_list(xs: list) -> list:
    """
    Function to flatten a list.

    Args:
        l (List[List[...]]):
            Nested lists that needs to be flattened.

    Returns:
        flatten list (list): flatted list.
    """
    out = list(chain(*xs))
    if len(out) and isinstance(out[0], list):
        out = flatten_list(out)
    return out


def get_step_inds(start: int, end: int, num: int):
    if num > (end - start):
        raise ValueError("num is larger than the number of total frames.")
    return np.around(np.linspace(start=start, stop=end, num=num, endpoint=False)).astype(int).tolist()


def _extract_frames(
    inds: List[int], video_path: str, max_size: int = 1920, color_base: str = "BGR", global_ind: int = 0
):
    # check video path
    if not is_video_file(video_path):
        raise TypeError(f"The video_path {video_path} is inappropriate.")

    # open cap
    cap = cv2.VideoCapture(video_path)
    # if start or end isn't specified lets assume 0
    start = inds[0]
    end = inds[-1] + 1
    # 設定cap frame 的啟始點
    cap.set(1, start)

    def _process_frame(frame):
        scale = max_size / max(frame.shape[:2])
        dst_h, dst_w = int(frame.shape[0] * scale), int(frame.shape[1] * scale)
        if scale < 1:
            frame = imresize(frame, (dst_h, dst_w))
        elif scale > 1:
            frame = imresize(frame, (dst_h, dst_w), interpolation=cv2.INTER_AREA)

        if color_base.upper() != "BGR":
            frame = imcvtcolor(frame, cvt_mode=f"BGR2{color_base}")
        return frame

    def _pickup_frame():
        for idx in range(start, end):
            _, frame = cap.read()
            if idx == inds[0]:
                inds.pop(0)
                # skip error frame
                if frame is None:
                    continue
                yield _process_frame(frame)

    # extract frames
    frames = list(_pickup_frame())

    # release cap
    cap.release()
    return frames, global_ind


def video2frames_v2(
    video_path: str,
    frame_per_sec: int = None,
    start_sec: float = 0,
    end_sec: float = None,
    n_threads: int = 8,
    max_size: int = 1920,
    color_base: str = "BGR",
) -> List[np.ndarray]:
    """
    Extracts the frames from a video using ray
    Inputs:
        video_path (str):
            path to the video.
        frame_per_sec (int, Optional):
            the number of extracting frames per sec.
            If None, all frames will be extracted.
        start_sec (int):
            the start second for frame extraction.
        end_sec (int):
            the end second for frame extraction.
        n_threads (int):
            the number of threads.
        max_size (int):
            max resolution of extracted frames.
        color_base (str):
            RGB or BGR color. Defaults to 'BGR'.
    Return:
        frames (list)
            [frame1, frame2, None, ...] or [frame1, frame2, ...,] else []
    """
    if not is_video_file(video_path):
        raise TypeError(f"The video_path {video_path} is inappropriate.")

    # get total_frames frames of video
    cap = cv2.VideoCapture(str(video_path))
    total_frames = round(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    if total_frames == 0 or fps == 0:
        return []

    frame_per_sec = fps if frame_per_sec is None else frame_per_sec
    total_sec = total_frames / fps
    # get frame inds
    end_sec = total_sec if end_sec is None or end_sec > total_sec else end_sec
    if start_sec > end_sec:
        raise ValueError(f"The start_sec should less than end_sec. {end_sec}")

    total_sec = end_sec - start_sec
    start_frame = round(start_sec * fps)
    end_frame = round(end_sec * fps)
    num = round(total_sec * frame_per_sec)
    frame_inds = get_step_inds(start_frame, end_frame, num)

    out_frames = []
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        ## -----start process---- ##
        # split the frames into chunk lists
        divide_size = round(len(frame_inds) / n_threads) + 1
        frame_inds_list = [frame_inds[i * divide_size : (i + 1) * divide_size] for i in range(n_threads)]
        future_to_frames = {
            executor.submit(_extract_frames, inds, video_path, max_size, color_base, i): inds
            for i, inds in enumerate(frame_inds_list)
        }
        out_frames = [[] for _ in range(n_threads)]

        for future in as_completed(future_to_frames):
            frames = future_to_frames[future]
            try:
                frames, global_ind = future.result()
                out_frames[global_ind] = frames
            except Exception as e:
                print(f"{frames} generated an exception: {e}")

        out_frames = flatten_list(out_frames)

        return out_frames
