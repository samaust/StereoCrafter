import cv2
import torch
from decord import VideoReader, cpu


def read_video_opencv(input_video_path: str):
    """
    Reads a video from disk
    """
    video_reader = VideoReader(input_video_path, ctx=cpu(0))
    frame_indices = list(range(len(video_reader)))
    frames = video_reader.get_batch(frame_indices)
    # [t,h,w,c]
    frames = torch.tensor(frames.asnumpy()).float()
    frames = frames / 255.0

    return frames


def read_video_opencv_four(input_video_path: str):
    """
    Reads a video from disk containing 4 videos in a grid
    """
    frames = read_video_opencv(input_video_path)
    height, width = frames.shape[1] // 2, frames.shape[2] // 2
    frames_left = frames[:, :height, :width, :]
    frames_mask = frames[:, height:, :width, :]
    frames_warped = frames[:, height:, width:, :]

    return (
        frames_left,
        frames_mask,
        frames_warped,
    )


def write_video_opencv(input_frames, fps, output_video_path):
    """
    Writes a video to disk
    """

    num_frames = len(input_frames)
    height, width, _ = input_frames[0].shape

    out = cv2.VideoWriter(
        output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    for i in range(num_frames):
        out.write(input_frames[i, :, :, ::-1])

    out.release()
