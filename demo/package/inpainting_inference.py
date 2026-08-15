"""Minimal package API example for selectable inpainting backends."""

from stereocrafter.inference.inpainting import create_inpainter
from stereocrafter.utils import read_video_opencv_four


def inpaint_video_frames(
    input_video_path: str,
    *,
    backend: str = "wan_vace",
    base_model: str | None = None,
    inpainting_model: str | None = None,
    **inference_options,
):
    frames_left, frames_mask, frames_warped = read_video_opencv_four(input_video_path)
    model_options = {
        key: value
        for key, value in {
            "base_model": base_model,
            "inpainting_model": inpainting_model,
        }.items()
        if value is not None
    }
    inpainter = create_inpainter(backend, **model_options)
    frames_right = inpainter.inpaint(
        frames_warped,
        frames_mask,
        **inference_options,
    )
    return frames_left[: len(frames_right)], frames_right
