import numpy as np
import torch
from diffusers import (
    AutoencoderKLTemporalDecoder,
    UNetSpatioTemporalConditionModel,
)
from transformers import CLIPVisionModelWithProjection

from stereocrafter.pipelines.stereo_video_inpainting import (
    StableVideoDiffusionInpaintingPipeline,
    _resize_with_antialiasing,
    tensor2vid,
)


def blend_h(a: torch.Tensor, b: torch.Tensor, overlap_size: int) -> torch.Tensor:
    weight_b = (torch.arange(overlap_size).view(1, 1, 1, -1) / overlap_size).to(
        b.device
    )
    b[:, :, :, :overlap_size] = (1 - weight_b) * a[
        :, :, :, -overlap_size:
    ] + weight_b * b[:, :, :, :overlap_size]
    return b


def blend_v(a: torch.Tensor, b: torch.Tensor, overlap_size: int) -> torch.Tensor:
    weight_b = (torch.arange(overlap_size).view(1, 1, -1, 1) / overlap_size).to(
        b.device
    )
    b[:, :, :overlap_size, :] = (1 - weight_b) * a[
        :, :, -overlap_size:, :
    ] + weight_b * b[:, :, :overlap_size, :]
    return b


def spatial_tiled_process(
    cond_frames,
    mask_frames,
    process_func,
    tile_num,
    spatial_n_compress=8,
    **kargs,
):
    height = cond_frames.shape[2]
    width = cond_frames.shape[3]

    tile_overlap = (128, 128)
    tile_size = (
        int((height + tile_overlap[0] * (tile_num - 1)) / tile_num),
        int((width + tile_overlap[1] * (tile_num - 1)) / tile_num),
    )
    tile_stride = ((tile_size[0] - tile_overlap[0]), (tile_size[1] - tile_overlap[1]))

    cols = []
    for i in range(0, tile_num):
        rows = []
        for j in range(0, tile_num):
            cond_tile = cond_frames[
                :,
                :,
                i * tile_stride[0] : i * tile_stride[0] + tile_size[0],
                j * tile_stride[1] : j * tile_stride[1] + tile_size[1],
            ]
            mask_tile = mask_frames[
                :,
                :,
                i * tile_stride[0] : i * tile_stride[0] + tile_size[0],
                j * tile_stride[1] : j * tile_stride[1] + tile_size[1],
            ]

            tile = process_func(
                frames=cond_tile,
                frames_mask=mask_tile,
                height=cond_tile.shape[2],
                width=cond_tile.shape[3],
                num_frames=len(cond_tile),
                output_type="latent",
                **kargs,
            ).frames[0]

            rows.append(tile)
        cols.append(rows)

    latent_stride = (
        tile_stride[0] // spatial_n_compress,
        tile_stride[1] // spatial_n_compress,
    )
    latent_overlap = (
        tile_overlap[0] // spatial_n_compress,
        tile_overlap[1] // spatial_n_compress,
    )

    results_cols = []
    for i, rows in enumerate(cols):
        results_rows = []
        for j, tile in enumerate(rows):
            if i > 0:
                tile = blend_v(cols[i - 1][j], tile, latent_overlap[0])
            if j > 0:
                tile = blend_h(rows[j - 1], tile, latent_overlap[1])
            results_rows.append(tile)
        results_cols.append(results_rows)

    pixels = []
    for i, rows in enumerate(results_cols):
        for j, tile in enumerate(rows):
            if i < len(results_cols) - 1:
                tile = tile[:, :, : latent_stride[0], :]
            if j < len(rows) - 1:
                tile = tile[:, :, :, : latent_stride[1]]
            rows[j] = tile
        pixels.append(torch.cat(rows, dim=3))
    x = torch.cat(pixels, dim=2)
    return x


def parse_resize_size(
    resize_size: tuple[int, int],
) -> tuple[int, int]:
    """
    Args:
        resize_size: A tuple of two ints (H, W).

    Returns:
        A tuple of two ints (H, W).

    Throws:
        ValueError if resize_size is invalid.
    """
    if not isinstance(resize_size, tuple):
        raise ValueError("Image size can only be a tuple of (H, W)")
    if len(resize_size) != 2:
        raise ValueError("Image size can only be a tuple of (H, W)")
    if not all(i > 0 for i in resize_size):
        raise ValueError("Image sizes must be greater than 0; got %d, %d" % resize_size)
    if not all(isinstance(i, int) for i in resize_size):
        raise ValueError("Image sizes must be integers; got %f, %f" % resize_size)
    return resize_size


def load_models(
    pre_trained_path: str = "stabilityai/stable-video-diffusion-img2vid-xt-1-1",
    unet_path: str = "TencentARC/StereoCrafter",
    *,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.float16,
) -> StableVideoDiffusionInpaintingPipeline:
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        pre_trained_path,
        subfolder="image_encoder",
        variant="fp16",
        torch_dtype=dtype,
    )

    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        pre_trained_path, subfolder="vae", variant="fp16", torch_dtype=dtype
    )

    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        unet_path,
        # variant="fp16",
        torch_dtype=dtype,
    )

    image_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)

    pipeline = StableVideoDiffusionInpaintingPipeline.from_pretrained(
        pre_trained_path,
        image_encoder=image_encoder,
        vae=vae,
        unet=unet,
        torch_dtype=dtype,
    )
    pipeline = pipeline.to(device)

    return pipeline


def tiled_inpaint(
    frames_warped: torch.Tensor,
    frames_mask: torch.Tensor,
    pipeline: StableVideoDiffusionInpaintingPipeline,
    resize_size: tuple[int, int] | None = None,
    fps: int = 16,
    frames_chunk: int = 25,
    overlap: int = 3,
    tile_num: int = 1,
    num_inference_steps: int = 8,
) -> torch.Tensor:
    """
    Inpaint all frames using tile_num tiles.

    Suggested settings
        frames resolution=(1024, 576) or (576, 1024)
        frames_chunk=25
        overlap=3
        tile_num=1

        frames resolution=(1920, 1024) or (1024, 1920)
        frames_chunk=25
        overlap=3
        tile_num=2
    """
    if frames_warped.ndim != 4 or frames_mask.ndim != 4:
        raise ValueError("frames_warped and frames_mask must use [T, H, W, C] layout")
    if frames_warped.shape[:3] != frames_mask.shape[:3]:
        raise ValueError("frames_warped and frames_mask must share T, H, and W")
    if frames_chunk <= 0 or not 0 <= overlap < frames_chunk:
        raise ValueError("overlap must be non-negative and less than frames_chunk")
    if tile_num <= 0 or num_inference_steps <= 0:
        raise ValueError("tile_num and num_inference_steps must be positive")

    num_frames_warped = frames_warped.shape[0]
    num_frames_mask = frames_mask.shape[0]

    if num_frames_warped != num_frames_mask:
        raise ValueError(
            "frames_warped and frames_mask must have the same number of frames"
        )

    # Remove alpha channel from frames_warped
    print(f"frames_warped.shape: {frames_warped.shape}")
    num_frames_warped_channels = frames_warped.shape[3]
    if num_frames_warped_channels == 4:
        # Alpha is intentionally dropped; inpainting currently produces RGB output.
        frames_warped = frames_warped[:, :, :, :3]

    # [t,h,w,c] -> [t,c,h,w]
    frames_mask = frames_mask.permute(0, 3, 1, 2).float()
    frames_warped = frames_warped.permute(0, 3, 1, 2).float()

    # Crop to multiple of 128
    # height_warped, width_warped = frames_warped.shape[2], frames_warped.shape[3]
    # height_warped = height_warped // 128 * 128
    # width_warped = width_warped // 128 * 128
    # frames_mask = frames_mask[:, :, 0:height_warped, 0:width_warped]
    # frames_warped = frames_warped[:, :, 0:height_warped, 0:width_warped]

    if resize_size is not None:
        resize_size = parse_resize_size(resize_size)
        resize_size_height = resize_size[0]
        resize_size_width = resize_size[1]

        assert resize_size_height == resize_size_height // 128 * 128, (
            "resize_size height must be divisible by 128"
        )
        assert resize_size_width == resize_size_width // 128 * 128, (
            "resize_size width must be divisible by 128"
        )

        print(f"frames_warped.shape: {frames_warped.shape}")
        print(f"frames_mask.shape: {frames_mask.shape}")

        # frames_warped = frames_warped * 2.0 - 1.0
        # frames_mask = frames_mask * 2.0 - 1.0

        frames_warped = _resize_with_antialiasing(frames_warped, resize_size).clamp(
            0, 1
        )
        frames_mask = _resize_with_antialiasing(frames_mask, resize_size).clamp(0, 1)

        # frames_warped = (frames_warped + 1.0) / 2.0
        # frames_mask = (frames_mask + 1.0) / 2.0

        print(f"frames_warped.shape: {frames_warped.shape}")
        print(f"frames_mask.shape: {frames_mask.shape}")

    height_warped, width_warped = frames_warped.shape[2], frames_warped.shape[3]
    height_mask, width_mask = frames_mask.shape[2], frames_mask.shape[3]

    assert height_mask == height_warped, (
        "frames_warped and frames_mask must have same height"
    )
    assert width_mask == width_warped, (
        "frames_warped and frames_mask must have same width"
    )

    assert height_warped == (height_warped // 128 * 128), (
        "height must be divisible by 128. height is %d" % height_warped
    )
    assert width_warped == (width_warped // 128 * 128), (
        "width must be divisible by 128. width is %d" % width_warped
    )

    # TODO : maybe move to depth splatting
    max_values, _ = torch.max(frames_mask, dim=1)
    max_expanded = torch.unsqueeze(max_values, dim=1)
    frames_mask = max_expanded.repeat(1, 3, 1, 1)
    frames_mask_rgb = torch.where(
        frames_mask > 0.25,
        torch.tensor(1.0, dtype=frames_mask.dtype, device=frames_mask.device),
        frames_mask,
    )
    frames_mask = frames_mask_rgb.mean(dim=1, keepdim=True)

    print(f"frames_mask.shape: {frames_mask.shape}")

    results = []
    generated = None
    for i in range(0, num_frames_warped, frames_chunk - overlap):
        if i + overlap >= frames_warped.shape[0]:
            break

        if generated is not None and i + frames_chunk > frames_warped.shape[0]:
            cur_i = max(frames_warped.shape[0] + overlap - frames_chunk, 0)
            cur_overlap = i - cur_i + overlap
        else:
            cur_i = i
            cur_overlap = overlap

        input_frames_i = frames_warped[cur_i : cur_i + frames_chunk].clone()
        mask_frames_i = frames_mask[cur_i : cur_i + frames_chunk]

        if generated is not None:
            try:
                input_frames_i[:cur_overlap] = generated[-cur_overlap:]
            except Exception as e:
                print(e)
                print(
                    f"i: {i}, cur_i: {cur_i}, cur_overlap: {cur_overlap}, input_frames_i: {input_frames_i.shape}, generated: {generated.shape}"
                )

        video_latents = spatial_tiled_process(
            input_frames_i,
            mask_frames_i,
            pipeline,
            tile_num,
            spatial_n_compress=8,
            min_guidance_scale=1.01,
            max_guidance_scale=1.01,
            decode_chunk_size=8,
            fps=fps,
            motion_bucket_id=127,
            noise_aug_strength=0.0,
            num_inference_steps=num_inference_steps,
        )

        video_latents = video_latents.unsqueeze(0)
        if video_latents.dtype == torch.float16:
            pipeline.vae.to(dtype=torch.float16)

        video_frames = pipeline.decode_latents(
            video_latents, num_frames=video_latents.shape[1], decode_chunk_size=2
        )
        video_frames = tensor2vid(
            video_frames, pipeline.image_processor, output_type="pil"
        )[0]

        for j in range(len(video_frames)):
            img = video_frames[j]
            video_frames[j] = (
                torch.tensor(np.array(img)).permute(2, 0, 1).to(dtype=torch.float32)
                / 255.0
            )
        generated = torch.stack(video_frames)
        if i != 0:
            generated = generated[cur_overlap:]
        results.append(generated)

    frames_output = torch.cat(results, dim=0).permute(0, 2, 3, 1)
    # frames_warped = frames_warped.permute(0, 2, 3, 1)
    # frames_mask_rgb = frames_mask_rgb.permute(0, 2, 3, 1)

    print(f"frames_output.shape: {frames_output.shape}")

    return frames_output


class SvdInpainter:
    """Stable Video Diffusion backend using the original StereoCrafter UNet."""

    name = "svd"

    def __init__(
        self,
        base_model: str = "stabilityai/stable-video-diffusion-img2vid-xt-1-1",
        inpainting_model: str = "TencentARC/StereoCrafter",
        *,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float16,
        pipeline: StableVideoDiffusionInpaintingPipeline | None = None,
    ) -> None:
        self.pipeline = pipeline or load_models(
            pre_trained_path=base_model,
            unet_path=inpainting_model,
            device=device,
            dtype=dtype,
        )

    def inpaint(
        self,
        frames_warped: torch.Tensor,
        frames_mask: torch.Tensor,
        **options,
    ) -> torch.Tensor:
        return tiled_inpaint(
            frames_warped,
            frames_mask,
            self.pipeline,
            **options,
        )
