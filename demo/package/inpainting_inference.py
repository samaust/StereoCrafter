import gc
import os

import torch

from stereocrafter.inference.inpainting import load_models, tiled_inpaint
from stereocrafter.pipelines.stereo_video_inpainting import _resize_with_antialiasing
from stereocrafter.utils import read_video_opencv_four, write_video_opencv


def main(input_video_path, save_dir, output_video_name, output_resize_size, **kwargs):
    # Load videos
    frames_left, frames_mask, frames_warped = read_video_opencv_four(input_video_path)

    # Load models
    pipeline = load_models()

    # Inpaint using tiles
    frames_output = tiled_inpaint(frames_warped, frames_mask, pipeline, **kwargs)

    # Release memory
    del frames_warped, frames_mask, pipeline
    # torch.cuda.empty_cache()
    gc.collect()

    print(f"frames_left.shape: {frames_left.shape}")
    print(f"frames_output.shape: {frames_output.shape}")

    if output_resize_size is None:
        # Don't resize
        assert frames_left.shape == frames_output.shape, (
            "left and right frames must have same shape"
        )
    else:
        # Resize
        if frames_left.shape != frames_output.shape:
            frames_left = frames_left.permute(0, 3, 1, 2).float()
            frames_left = _resize_with_antialiasing(
                frames_left, output_resize_size
            ).clamp(0, 1)
            frames_left = frames_left.permute(0, 2, 3, 1)

        frames_output = frames_output.permute(0, 3, 1, 2).float()
        frames_output = _resize_with_antialiasing(
            frames_output, output_resize_size
        ).clamp(0, 1)
        frames_output = frames_output.permute(0, 2, 3, 1)

    print(f"frames_left.shape: {frames_left.shape}")
    print(f"frames_output.shape: {frames_output.shape}")

    # Concatenate left and right videos
    frames_sbs = torch.cat([frames_left, frames_output], dim=2)

    print(f"frames_sbs.shape: {frames_sbs.shape}")

    # Convert float to uint8
    frames_sbs = (frames_sbs * 255).to(dtype=torch.uint8).cpu().numpy()

    # Save video file
    fps = kwargs["fps"]
    frames_sbs_path = os.path.join(
        save_dir,
        f"{output_video_name}_f{kwargs['fps']}_fc{kwargs['frames_chunk']}_o{kwargs['overlap']}_tn{kwargs['tile_num']}_nis{kwargs['num_inference_steps']}.mp4",
    )
    write_video_opencv(frames_sbs, fps, frames_sbs_path)


if __name__ == "__main__":
    # Parameters
    input_video_path = r"./outputs/tt_splatting_results.mp4"
    save_dir = r"./outputs"
    output_video_name = "tt"
    output_resize_size = (
        720,
        480,
    )

    # kwargs list to experiment with the settings
    kwargs_list = [
        {
            "resize_size": (
                1920,
                1024,
            ),
            "fps": 16,
            "frames_chunk": 25,
            "overlap": 3,
            "tile_num": 2,
            "num_inference_steps": 1,
        },
        {
            "resize_size": (
                1920,
                1024,
            ),
            "fps": 16,
            "frames_chunk": 25,
            "overlap": 3,
            "tile_num": 2,
            "num_inference_steps": 2,
        },
        {
            "resize_size": (
                1920,
                1024,
            ),
            "fps": 16,
            "frames_chunk": 25,
            "overlap": 3,
            "tile_num": 2,
            "num_inference_steps": 3,
        },
        {
            "resize_size": (
                1920,
                1024,
            ),
            "fps": 16,
            "frames_chunk": 25,
            "overlap": 3,
            "tile_num": 2,
            "num_inference_steps": 4,
        },
        {
            "resize_size": (
                1920,
                1024,
            ),
            "fps": 16,
            "frames_chunk": 25,
            "overlap": 3,
            "tile_num": 2,
            "num_inference_steps": 5,
        },
        {
            "resize_size": (
                1920,
                1024,
            ),
            "fps": 16,
            "frames_chunk": 25,
            "overlap": 3,
            "tile_num": 2,
            "num_inference_steps": 6,
        },
        {
            "resize_size": (
                1920,
                1024,
            ),
            "fps": 16,
            "frames_chunk": 25,
            "overlap": 3,
            "tile_num": 2,
            "num_inference_steps": 7,
        },
        {
            "resize_size": (
                1920,
                1024,
            ),
            "fps": 16,
            "frames_chunk": 25,
            "overlap": 3,
            "tile_num": 2,
            "num_inference_steps": 8,
        },
        {
            "resize_size": (
                1920,
                1024,
            ),
            "fps": 16,
            "frames_chunk": 25,
            "overlap": 3,
            "tile_num": 2,
            "num_inference_steps": 9,
        },
        {
            "resize_size": (
                1920,
                1024,
            ),
            "fps": 16,
            "frames_chunk": 25,
            "overlap": 3,
            "tile_num": 2,
            "num_inference_steps": 10,
        },
    ]

    for kwargs in kwargs_list:
        main(
            input_video_path, save_dir, output_video_name, output_resize_size, **kwargs
        )
