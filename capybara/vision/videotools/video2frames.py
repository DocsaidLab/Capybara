import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

__all__ = ["is_video_file", "video2frames"]

VIDEO_SUFFIX = [".MOV", ".MP4", ".AVI", ".WEBM", ".3GP", ".MKV"]


def is_video_file(x: Any) -> bool:
    x = Path(x)
    cond1 = x.exists()
    cond2 = x.suffix.upper() in VIDEO_SUFFIX
    return cond1 and cond2


def video2frames(
    video_path: str | Path,
    frame_per_sec: int | None = None,
) -> list[np.ndarray]:
    """
    Extracts the frames from a video using ray
    Inputs:
        video_path (str): Path to the video.
        frame_per_sec (int, Optional): The number of extracting frames per sec.
            If None, all frames will be extracted.

    Returns:
        frames (List[np.ndarray]): A list of frames.
    """
    if not is_video_file(video_path):
        raise TypeError(f"The video_path {video_path} is inappropriate.")

    # get total_frames frames of video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return []

    # Get the original FPS of the video
    original_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

    # Calculate the interval for frame extraction
    if frame_per_sec is None:
        interval = 1
    else:
        frame_per_sec_i = int(frame_per_sec)
        if frame_per_sec_i <= 0:
            raise ValueError("frame_per_sec must be > 0.")
        if not math.isfinite(original_fps) or original_fps <= 0:
            interval = 1
        else:
            interval = max(1, int(original_fps / frame_per_sec_i))

    frames = []
    index = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Extract the frame if it's on the specified interval
        if index % interval == 0:
            frames.append(frame)
        index += 1
    cap.release()

    return frames
