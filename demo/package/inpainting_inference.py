"""Minimal package API example for selectable inpainting backends."""

from stereocrafter.inference.inpainting import create_inpainter
from stereocrafter.utils import read_video_opencv_four


def inpaint_video_frames(
    input_video_path: str,
    *,
    backend: str = "wan_vace",
    base_model: str | None = None,
    inpainting_model: str | None = None,
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
    frames_left, frames_mask, frames_warped = read_video_opencv_four(input_video_path)
    model_options = {
        key: value
        for key, value in {
            "base_model": base_model,
            "inpainting_model": inpainting_model,
        }.items()
        if value is not None
    }
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
    frames_right = inpainter.inpaint(
        frames_warped,
        frames_mask,
        **inference_options,
    )
    return frames_left[: len(frames_right)], frames_right
