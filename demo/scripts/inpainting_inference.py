import os

import numpy as np
import torch
from decord import VideoReader, cpu
from diffusers.utils import export_to_video
from fire import Fire

from stereocrafter.inference.inpainting import create_inpainter
from stereocrafter.utils import read_video_opencv_four


def _export_stereo_videos(
    frames_left: torch.Tensor,
    frames_right: torch.Tensor,
    save_dir: str,
    video_name: str,
    fps: int,
) -> tuple[str, str]:
    if frames_left.shape != frames_right.shape:
        raise ValueError(
            "Left- and right-eye output shapes differ: "
            f"{tuple(frames_left.shape)} != {tuple(frames_right.shape)}"
        )

    os.makedirs(save_dir, exist_ok=True)
    left = frames_left.cpu().float().numpy()
    right = frames_right.cpu().float().numpy()

    sbs = np.concatenate([left, right], axis=2)
    sbs_path = os.path.join(save_dir, f"{video_name}_sbs.mp4")
    export_to_video(list(sbs), sbs_path, fps=fps)

    anaglyph_left = left.copy()
    anaglyph_right = right.copy()
    anaglyph_left[..., 1:] = 0
    anaglyph_right[..., 0] = 0
    anaglyph_path = os.path.join(save_dir, f"{video_name}_anaglyph.mp4")
    export_to_video(
        list(np.clip(anaglyph_left + anaglyph_right, 0, 1)),
        anaglyph_path,
        fps=fps,
    )
    return sbs_path, anaglyph_path


def main(
    input_video_path: str,
    save_dir: str,
    backend: str = "wan_vace",
    base_model: str | None = None,
    inpainting_model: str | None = None,
    video_name: str | None = None,
    fps: int | None = None,
    device: str | None = None,
    model_precision: str = "fp8",
    quantized_cache_dir: str | None = None,
    rebuild_quantized_cache: bool = False,
    text_encoder_device: str | None = None,
    vae_device: str | None = None,
    sequential_offload: bool = True,
    vae_tiling: bool = True,
    vae_tile_size: int = 256,
    vae_tile_stride: int = 192,
    **inference_options,
):
    """Inpaint a depth-splatting result with a selectable package backend."""

    frames_left, frames_mask, frames_warped = read_video_opencv_four(input_video_path)
    if fps is None:
        fps = int(round(VideoReader(input_video_path, ctx=cpu(0)).get_avg_fps()))

    model_options = {}
    if base_model is not None:
        model_options["base_model"] = base_model
    if inpainting_model is not None:
        model_options["inpainting_model"] = inpainting_model
    if device is not None:
        model_options["device"] = device
    if backend == "wan_vace":
        model_options.update(
            model_precision=model_precision,
            rebuild_quantized_cache=rebuild_quantized_cache,
            sequential_offload=sequential_offload,
            vae_tiling=vae_tiling,
            vae_tile_size=vae_tile_size,
            vae_tile_stride=vae_tile_stride,
        )
        if quantized_cache_dir is not None:
            model_options["quantized_cache_dir"] = quantized_cache_dir
        if text_encoder_device is not None:
            model_options["text_encoder_device"] = text_encoder_device
        if vae_device is not None:
            model_options["vae_device"] = vae_device

    inpainter = create_inpainter(backend, **model_options)
    if backend == "svd":
        if "frames_overlap" in inference_options:
            inference_options["overlap"] = inference_options.pop("frames_overlap")
        if "inference_steps" in inference_options:
            inference_options["num_inference_steps"] = inference_options.pop(
                "inference_steps"
            )
        inference_options.setdefault("fps", fps)

    frames_right = inpainter.inpaint(
        frames_warped,
        frames_mask,
        **inference_options,
    )
    frames_left = frames_left[: len(frames_right)]
    video_name = video_name or (
        os.path.splitext(os.path.basename(input_video_path))[0].replace(
            "_splatting_results", ""
        )
        + "_inpainting_results"
    )
    paths = _export_stereo_videos(
        frames_left,
        frames_right,
        save_dir,
        video_name,
        fps,
    )
    print(f"Saved side-by-side video to {paths[0]}")
    print(f"Saved anaglyph video to {paths[1]}")


if __name__ == "__main__":
    Fire(main)
