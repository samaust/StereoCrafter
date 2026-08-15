import html
import math
import re
from typing import Any

import ftfy
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKLWan, WanVACETransformer3DModel
from diffusers.video_processor import VideoProcessor
from PIL import Image
from transformers import AutoTokenizer, UMT5EncoderModel


class FlowMatchScheduler:
    def __init__(
        self,
    ):
        self.set_timesteps_fn = FlowMatchScheduler.set_timesteps_wan
        self.num_train_timesteps = 1000

    @staticmethod
    def set_timesteps_wan(num_inference_steps=100, denoising_strength=1.0, shift=None):
        sigma_min = 0.0
        sigma_max = 1.0
        shift = 5 if shift is None else shift
        num_train_timesteps = 1000
        sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength
        sigmas = torch.linspace(sigma_start, sigma_min, num_inference_steps + 1)[:-1]
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        timesteps = sigmas * num_train_timesteps
        return sigmas, timesteps

    def set_timesteps(self, num_inference_steps=100, denoising_strength=1.0, **kwargs):
        self.sigmas, self.timesteps = self.set_timesteps_fn(
            num_inference_steps=num_inference_steps,
            denoising_strength=denoising_strength,
            **kwargs,
        )

    def step(self, model_output, timestep, sample, to_final=False, **kwargs):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        if to_final or timestep_id + 1 >= len(self.timesteps):
            sigma_ = 0
        else:
            sigma_ = self.sigmas[timestep_id + 1]
        prev_sample = sample + model_output * (sigma_ - sigma)
        return prev_sample


def encode_vae_mode(vae, x):
    dist = vae.encode(x).latent_dist
    return dist.mode() if hasattr(dist, "mode") else dist.mean


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def prompt_clean(text):
    text = whitespace_clean(basic_clean(text))
    return text


def get_t5_prompt_embeds(
    prompt=None,
    num_videos_per_prompt=1,
    max_sequence_length=226,
    device=None,
    dtype=None,
    tokenizer=None,
    text_encoder=None,
):
    # device = device or self._execution_device
    # dtype = dtype or self.text_encoder.dtype

    prompt = [prompt] if isinstance(prompt, str) else prompt
    prompt = [prompt_clean(u) for u in prompt]
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
    seq_lens = mask.gt(0).sum(dim=1).long()

    prompt_embeds = text_encoder(
        text_input_ids.to(device), mask.to(device)
    ).last_hidden_state
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
    prompt_embeds = torch.stack(
        [
            torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))])
            for u in prompt_embeds
        ],
        dim=0,
    )

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

    return prompt_embeds


def encode_prompt(
    prompt,
    negative_prompt=None,
    do_classifier_free_guidance=True,
    num_videos_per_prompt=1,
    prompt_embeds=None,
    negative_prompt_embeds=None,
    max_sequence_length=226,
    device=None,
    dtype=None,
    tokenizer=None,
    text_encoder=None,
):
    r"""
    Encodes the prompt into text encoder hidden states.

    Args:
        prompt (`str` or `List[str]`, *optional*):
            prompt to be encoded
        negative_prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts not to guide the image generation. If not defined, one has to pass
            `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
            less than `1`).
        do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
            Whether to use classifier free guidance or not.
        num_videos_per_prompt (`int`, *optional*, defaults to 1):
            Number of videos that should be generated per prompt. torch device to place the resulting embeddings on
        prompt_embeds (`torch.Tensor`, *optional*):
            Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
            provided, text embeddings will be generated from `prompt` input argument.
        negative_prompt_embeds (`torch.Tensor`, *optional*):
            Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
            weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
            argument.
        device: (`torch.device`, *optional*):
            torch device
        dtype: (`torch.dtype`, *optional*):
            torch dtype
    """
    # device = device or self._execution_device

    prompt = [prompt] if isinstance(prompt, str) else prompt
    if prompt is not None:
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    if prompt_embeds is None:
        prompt_embeds = get_t5_prompt_embeds(
            prompt=prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
        )

    if do_classifier_free_guidance and negative_prompt_embeds is None:
        negative_prompt = negative_prompt or ""
        negative_prompt = (
            batch_size * [negative_prompt]
            if isinstance(negative_prompt, str)
            else negative_prompt
        )

        if prompt is not None and type(prompt) is not type(negative_prompt):
            raise TypeError(
                f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                f" {type(prompt)}."
            )
        elif batch_size != len(negative_prompt):
            raise ValueError(
                f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                " the batch size of `prompt`."
            )

        negative_prompt_embeds = get_t5_prompt_embeds(
            prompt=negative_prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
        )

    return prompt_embeds, negative_prompt_embeds


def prepare_masks(
    mask: torch.Tensor,
    reference_images=None,
    # generator = None,
    transformer_patch_size=None,
    vae_scale_factor_temporal=None,
    vae_scale_factor_spatial=None,
) -> torch.Tensor:
    # if isinstance(generator, list):
    #     # TODO: support this
    #     raise ValueError("Passing a list of generators is not yet supported. This may be supported in the future.")

    if reference_images is None:
        # For each batch of video, we set no reference image (as one or more can be passed by user)
        reference_images = [[None] for _ in range(mask.shape[0])]
    else:
        if mask.shape[0] != len(reference_images):
            raise ValueError(
                f"Batch size of `mask` {mask.shape[0]} and length of `reference_images` {len(reference_images)} does not match."
            )

    # if mask.shape[0] != 1:
    #     # TODO: support this
    #     raise ValueError(
    #         "Generating with more than one video is not yet supported. This may be supported in the future."
    #     )

    # transformer_patch_size = (
    #     self.transformer.config.patch_size[1]
    #     if self.transformer is not None
    #     else self.transformer_2.config.patch_size[1]
    # )

    mask_list = []
    for mask_, reference_images_batch in zip(mask, reference_images):
        num_channels, num_frames, height, width = mask_.shape
        new_num_frames = (
            num_frames + vae_scale_factor_temporal - 1
        ) // vae_scale_factor_temporal
        new_height = (
            height
            // (vae_scale_factor_spatial * transformer_patch_size)
            * transformer_patch_size
        )
        new_width = (
            width
            // (vae_scale_factor_spatial * transformer_patch_size)
            * transformer_patch_size
        )
        mask_ = mask_[0, :, :, :]
        mask_ = mask_.view(
            num_frames,
            new_height,
            vae_scale_factor_spatial,
            new_width,
            vae_scale_factor_spatial,
        )
        mask_ = mask_.permute(2, 4, 0, 1, 3).flatten(
            0, 1
        )  # [8x8, num_frames, new_height, new_width]
        mask_ = torch.nn.functional.interpolate(
            mask_.unsqueeze(0),
            size=(new_num_frames, new_height, new_width),
            mode="nearest-exact",
        ).squeeze(0)
        num_ref_images = len(reference_images_batch)
        if num_ref_images > 0:
            mask_padding = torch.zeros_like(mask_[:, :num_ref_images, :, :])
            mask_ = torch.cat([mask_padding, mask_], dim=1)
        mask_list.append(mask_)
    return torch.stack(mask_list)


def preprocess_conditions(
    video=None,
    mask=None,
    reference_images=None,
    batch_size: int = 1,
    height: int = 480,
    width: int = 832,
    num_frames: int = 81,
    dtype=None,
    device=None,
    video_processor=None,
    base=None,
):
    if video is not None:
        # base = self.vae_scale_factor_spatial * (
        #     self.transformer.config.patch_size[1]
        #     if self.transformer is not None
        #     else self.transformer_2.config.patch_size[1]
        # )
        video_height, video_width = video_processor.get_default_height_width(video[0])

        if video_height * video_width > height * width:
            scale = min(width / video_width, height / video_height)
            video_height, video_width = (
                int(video_height * scale),
                int(video_width * scale),
            )

        if video_height % base != 0 or video_width % base != 0:
            # logger.warning(
            #     f"Video height and width should be divisible by {base}, but got {video_height} and {video_width}. "
            # )
            video_height = (video_height // base) * base
            video_width = (video_width // base) * base

        assert video_height * video_width <= height * width

        video = video_processor.preprocess_video(video, video_height, video_width)
        image_size = (
            video_height,
            video_width,
        )  # Use the height/width of video (with possible rescaling)
    else:
        video = torch.zeros(
            batch_size, 3, num_frames, height, width, dtype=dtype, device=device
        )
        image_size = (height, width)  # Use the height/width provider by user

    if mask is not None:
        mask = video_processor.preprocess_video(mask, image_size[0], image_size[1])
        mask = torch.clamp((mask + 1) / 2, min=0, max=1)
    else:
        mask = torch.ones_like(video)

    video = video.to(dtype=dtype, device=device)
    mask = mask.to(dtype=dtype, device=device)

    # Make a list of list of images where the outer list corresponds to video batch size and the inner list
    # corresponds to list of conditioning images per video
    if reference_images is None or isinstance(reference_images, Image.Image):
        reference_images = [[reference_images] for _ in range(video.shape[0])]
    elif isinstance(reference_images, (list, tuple)) and isinstance(
        next(iter(reference_images)), Image.Image
    ):
        reference_images = [reference_images]
    elif (
        isinstance(reference_images, (list, tuple))
        and isinstance(next(iter(reference_images)), list)
        and isinstance(next(iter(reference_images[0])), Image.Image)
    ):
        reference_images = reference_images
    else:
        raise ValueError(
            "`reference_images` has to be of type `PIL.Image.Image` or `list` of `PIL.Image.Image`, or "
            "`list` of `list` of `PIL.Image.Image`, but is {type(reference_images)}"
        )

    if video.shape[0] != len(reference_images):
        raise ValueError(
            f"Batch size of `video` {video.shape[0]} and length of `reference_images` {len(reference_images)} does not match."
        )

    ref_images_lengths = [
        len(reference_images_batch) for reference_images_batch in reference_images
    ]
    if any(length != ref_images_lengths[0] for length in ref_images_lengths):
        raise ValueError(
            f"All batches of `reference_images` should have the same length, but got {ref_images_lengths}. Support for this "
            "may be added in the future."
        )

    reference_images_preprocessed = []
    for i, reference_images_batch in enumerate(reference_images):
        preprocessed_images = []
        for j, image in enumerate(reference_images_batch):
            if image is None:
                continue
            image = video_processor.preprocess(image, None, None)
            img_height, img_width = image.shape[-2:]
            scale = min(image_size[0] / img_height, image_size[1] / img_width)
            new_height, new_width = int(img_height * scale), int(img_width * scale)
            resized_image = torch.nn.functional.interpolate(
                image,
                size=(new_height, new_width),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)  # [C, H, W]
            top = (image_size[0] - new_height) // 2
            left = (image_size[1] - new_width) // 2
            canvas = torch.ones(3, *image_size, device=device, dtype=dtype)
            canvas[:, top : top + new_height, left : left + new_width] = resized_image
            preprocessed_images.append(canvas)
        reference_images_preprocessed.append(preprocessed_images)

    return video, mask, reference_images_preprocessed


def prepare_video_latents(
    video: torch.Tensor,
    mask: torch.Tensor,
    reference_images=None,
    device=None,
    vae=None,
) -> torch.Tensor:
    # device = device or self._execution_device

    # if isinstance(generator, list):
    #     # TODO: support this
    #     raise ValueError("Passing a list of generators is not yet supported. This may be supported in the future.")

    if reference_images is None:
        # For each batch of video, we set no re
        # ference image (as one or more can be passed by user)
        reference_images = [[None] for _ in range(video.shape[0])]
    else:
        if video.shape[0] != len(reference_images):
            raise ValueError(
                f"Batch size of `video` {video.shape[0]} and length of `reference_images` {len(reference_images)} does not match."
            )

    # if video.shape[0] != 1:
    #     # TODO: support this
    #     raise ValueError(
    #         "Generating with more than one video is not yet supported. This may be supported in the future."
    #     )

    vae_dtype = vae.dtype
    video = video.to(dtype=vae_dtype)

    latents_mean = torch.tensor(
        vae.config.latents_mean, device=device, dtype=torch.float32
    ).view(1, vae.config.z_dim, 1, 1, 1)
    latents_std = 1.0 / torch.tensor(
        vae.config.latents_std, device=device, dtype=torch.float32
    ).view(1, vae.config.z_dim, 1, 1, 1)

    if mask is None:
        # latents = retrieve_latents(vae.encode(video), generator, sample_mode="argmax").unbind(0)
        latents = encode_vae_mode(vae, video)
        latents = ((latents.float() - latents_mean) * latents_std).to(vae_dtype)
    else:
        mask = torch.where(mask > 0.5, 1.0, 0.0).to(dtype=vae_dtype)
        inactive = video * (1 - mask)
        reactive = video * mask
        # inactive = retrieve_latents(vae.encode(inactive), generator, sample_mode="argmax")
        inactive = encode_vae_mode(vae, inactive)
        # reactive = retrieve_latents(vae.encode(reactive), generator, sample_mode="argmax")
        reactive = encode_vae_mode(vae, reactive)
        inactive = ((inactive.float() - latents_mean) * latents_std).to(vae_dtype)
        reactive = ((reactive.float() - latents_mean) * latents_std).to(vae_dtype)
        latents = torch.cat([inactive, reactive], dim=1)

    latent_list = []
    for latent, reference_images_batch in zip(latents, reference_images):
        for reference_image in reference_images_batch:
            assert reference_image.ndim == 3
            reference_image = reference_image.to(dtype=vae_dtype)
            reference_image = reference_image[None, :, None, :, :]  # [1, C, 1, H, W]
            # reference_latent = retrieve_latents(vae.encode(reference_image), generator, sample_mode="argmax")
            reference_latent = vae.encode(reference_image).latent_dist.sample()
            reference_latent = (
                (reference_latent.float() - latents_mean) * latents_std
            ).to(vae_dtype)
            reference_latent = reference_latent.squeeze(0)  # [C, 1, H, W]
            reference_latent = torch.cat(
                [reference_latent, torch.zeros_like(reference_latent)], dim=0
            )
            latent = torch.cat([reference_latent.squeeze(0), latent], dim=1)
        latent_list.append(latent)

    return torch.stack(latent_list)


def blend_h(a: torch.Tensor, b: torch.Tensor, overlap_size: int) -> torch.Tensor:
    """Latents [B, C, F, H, W]"""
    weight_b = (torch.arange(overlap_size).view(1, 1, 1, 1, -1) / overlap_size).to(
        b.device, dtype=b.dtype
    )
    b[:, :, :, :, :overlap_size] = (1 - weight_b) * a[
        :, :, :, :, -overlap_size:
    ] + weight_b * b[:, :, :, :, :overlap_size]
    return b


def blend_v(a: torch.Tensor, b: torch.Tensor, overlap_size: int) -> torch.Tensor:
    """Latents [B, C, F, H, W]"""
    weight_b = (torch.arange(overlap_size).view(1, 1, 1, -1, 1) / overlap_size).to(
        b.device, dtype=b.dtype
    )
    b[:, :, :, :overlap_size, :] = (1 - weight_b) * a[
        :, :, :, -overlap_size:, :
    ] + weight_b * b[:, :, :, :overlap_size, :]
    return b


def run_wan_pipeline(
    cond_frames,
    mask_frames,
    prompt_embeds,
    transformer,
    vae,
    noise_scheduler,
    videoprocessor,
    vae_scale_factor_spatial,
    vae_scale_factor_temporal,
    transformer_patch_size,
    device,
    dtype,
    generator,
):
    """Run one Wan denoising pass."""
    #  cond_frames  [B, C, F, H, W]
    height, width = cond_frames.shape[3], cond_frames.shape[4]
    num_frames = cond_frames.shape[2]

    with torch.no_grad():
        # VideoProcessor  [B, F, C, H, W]
        # [B, C, F, H, W] -> [B, F, C, H, W]
        cond_frames_vp = cond_frames.permute(0, 2, 1, 3, 4)
        mask_frames_vp = mask_frames.permute(0, 2, 1, 3, 4)

        condition_video, mask, reference_images = preprocess_conditions(
            video=cond_frames_vp,
            mask=mask_frames_vp,
            reference_images=None,
            batch_size=1,
            height=height,
            width=width,
            num_frames=num_frames,
            dtype=dtype,
            device=device,
            video_processor=videoprocessor,
            base=vae_scale_factor_spatial * transformer_patch_size,
        )

        conditioning_latents = prepare_video_latents(
            condition_video, mask, reference_images, device, vae
        )
        mask_for_transformer = prepare_masks(
            mask,
            reference_images,
            transformer_patch_size,
            vae_scale_factor_temporal,
            vae_scale_factor_spatial,
        ).to(device, dtype=dtype)
        control_hidden_states = torch.cat(
            [conditioning_latents, mask_for_transformer], dim=1
        ).to(dtype)

    c = transformer.config.in_channels
    f = (num_frames - 1) // vae_scale_factor_temporal + 1
    h = height // vae_scale_factor_spatial
    w = width // vae_scale_factor_spatial

    latents = torch.randn(
        1, c, f, h, w, device=device, dtype=dtype, generator=generator
    )

    for t in noise_scheduler.timesteps:
        timestep_tensor = t.unsqueeze(0).to(device, dtype=dtype)
        with torch.no_grad():
            model_pred = transformer(
                hidden_states=latents,
                timestep=timestep_tensor,
                encoder_hidden_states=prompt_embeds,
                control_hidden_states=control_hidden_states,
                return_dict=False,
            )[0]
        latents = noise_scheduler.step(model_pred, t, latents)

    return latents


def spatial_tiled_process(
    cond_frames,
    mask_frames,
    tile_num,
    tile_overlap,
    prompt_embeds,
    transformer,
    vae,
    noise_scheduler,
    videoprocessor,
    vae_scale_factor_spatial,
    vae_scale_factor_temporal,
    transformer_patch_size,
    device,
    dtype,
    generator,
):
    """Run spatially tiled Wan inference."""
    if tile_num == 1:
        return run_wan_pipeline(
            cond_frames,
            mask_frames,
            prompt_embeds,
            transformer,
            vae,
            noise_scheduler,
            videoprocessor,
            vae_scale_factor_spatial,
            vae_scale_factor_temporal,
            transformer_patch_size,
            device,
            dtype,
            generator,
        )

    height = cond_frames.shape[3]
    width = cond_frames.shape[4]

    #  VAE  Transformer Patch  16
    base = vae_scale_factor_spatial * transformer_patch_size
    tile_size = (
        int((height + tile_overlap * (tile_num - 1)) / tile_num) // base * base,
        int((width + tile_overlap * (tile_num - 1)) / tile_num) // base * base,
    )
    tile_stride = (tile_size[0] - tile_overlap, tile_size[1] - tile_overlap)

    cols = []
    for i in range(tile_num):
        rows = []
        for j in range(tile_num):
            h_start = min(i * tile_stride[0], height - tile_size[0])
            w_start = min(j * tile_stride[1], width - tile_size[1])

            cond_tile = cond_frames[
                :,
                :,
                :,
                h_start : h_start + tile_size[0],
                w_start : w_start + tile_size[1],
            ]
            mask_tile = mask_frames[
                :,
                :,
                :,
                h_start : h_start + tile_size[0],
                w_start : w_start + tile_size[1],
            ]

            tile_latent = run_wan_pipeline(
                cond_tile,
                mask_tile,
                prompt_embeds,
                transformer,
                vae,
                noise_scheduler,
                videoprocessor,
                vae_scale_factor_spatial,
                vae_scale_factor_temporal,
                transformer_patch_size,
                device,
                dtype,
                generator,
            )
            rows.append(tile_latent)
        cols.append(rows)

    #  Latent  stride  overlap
    latent_stride = (
        tile_stride[0] // vae_scale_factor_spatial,
        tile_stride[1] // vae_scale_factor_spatial,
    )
    latent_overlap = (
        tile_overlap // vae_scale_factor_spatial,
        tile_overlap // vae_scale_factor_spatial,
    )

    #  Latents
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
                tile = tile[:, :, :, : latent_stride[0], :]
            if j < len(rows) - 1:
                tile = tile[:, :, :, :, : latent_stride[1]]
            rows[j] = tile
        pixels.append(torch.cat(rows, dim=4))

    return torch.cat(pixels, dim=3)


class WanVaceInpainter:
    """Wan VACE backend using the StereoCrafter2 transformer."""

    name = "wan_vace"

    def __init__(
        self,
        base_model: str = "Wan-AI/Wan2.1-VACE-14B-diffusers",
        inpainting_model: str = "TencentARC/StereoCrafter2",
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
        tokenizer: Any | None = None,
        text_encoder: Any | None = None,
        vae: Any | None = None,
        transformer: Any | None = None,
    ) -> None:
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.dtype = dtype
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(
            base_model, subfolder="tokenizer"
        )
        self.text_encoder = text_encoder or UMT5EncoderModel.from_pretrained(
            base_model, subfolder="text_encoder", torch_dtype=dtype
        ).to(self.device)
        self.vae = vae or AutoencoderKLWan.from_pretrained(
            base_model, subfolder="vae", torch_dtype=dtype
        ).to(self.device)
        self.transformer = transformer or WanVACETransformer3DModel.from_pretrained(
            inpainting_model, torch_dtype=dtype
        ).to(self.device)
        for model in (self.text_encoder, self.vae, self.transformer):
            model.eval()
            model.requires_grad_(False)

        self.video_processor = VideoProcessor(
            vae_scale_factor=self.vae.config.scale_factor_spatial
        )
        self.transformer_patch_size = self.transformer.config.patch_size[1]
        self.vae_scale_factor_temporal = 2 ** sum(self.vae.temperal_downsample)
        self.vae_scale_factor_spatial = 2 ** len(self.vae.temperal_downsample)

    def _prompt_embeds(self, prompt: str) -> torch.Tensor:
        with torch.no_grad():
            embeds, _ = encode_prompt(
                [prompt],
                do_classifier_free_guidance=False,
                max_sequence_length=226,
                device=self.device,
                dtype=self.dtype,
                tokenizer=self.tokenizer,
                text_encoder=self.text_encoder,
            )
        return embeds

    def inpaint(
        self,
        frames_warped: torch.Tensor,
        frames_mask: torch.Tensor,
        *,
        frames_chunk: int = 81,
        frames_overlap: int = 10,
        tile_overlap: int = 128,
        tile_num: int = 2,
        inference_steps: int = 10,
        seed: int | None = 0,
        prompt: str = "",
        **_: Any,
    ) -> torch.Tensor:
        _validate_video_inputs(
            frames_warped,
            frames_mask,
            frames_chunk=frames_chunk,
            frames_overlap=frames_overlap,
            tile_num=tile_num,
            tile_overlap=tile_overlap,
            inference_steps=inference_steps,
        )
        frames = frames_warped[..., :3].permute(3, 0, 1, 2).unsqueeze(0).float()
        masks = frames_mask.permute(3, 0, 1, 2).unsqueeze(0).float()
        if masks.shape[1] not in (1, frames.shape[1]):
            raise ValueError("frames_mask must have one channel or match RGB channels")
        frames = frames.to(self.device)
        masks = masks.to(self.device)
        frames = frames * (1.0 - masks) + 0.5 * masks

        base = self.vae_scale_factor_spatial * self.transformer_patch_size
        original_height, original_width = frames.shape[-2:]
        tile_size_h = (
            math.ceil(
                ((original_height + tile_overlap * (tile_num - 1)) / tile_num) / base
            )
            * base
        )
        tile_size_w = (
            math.ceil(
                ((original_width + tile_overlap * (tile_num - 1)) / tile_num) / base
            )
            * base
        )
        target_height = (tile_size_h - tile_overlap) * (tile_num - 1) + tile_size_h
        target_width = (tile_size_w - tile_overlap) * (tile_num - 1) + tile_size_w
        pad_height = target_height - original_height
        pad_width = target_width - original_width
        if pad_height < 0 or pad_width < 0:
            raise ValueError("tile_overlap is too large for the requested tile layout")
        if pad_height or pad_width:
            frames_4d = frames[0].permute(1, 0, 2, 3)
            frames_4d = F.pad(
                frames_4d, (0, pad_width, 0, pad_height), mode="replicate"
            )
            frames = frames_4d.permute(1, 0, 2, 3).unsqueeze(0)
            masks = F.pad(
                masks, (0, pad_width, 0, pad_height), mode="constant", value=0
            )

        scheduler = FlowMatchScheduler()
        scheduler.set_timesteps(
            num_inference_steps=inference_steps, denoising_strength=1.0
        )
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        prompt_embeds = self._prompt_embeds(prompt)

        total_frames = frames.shape[2]
        generated_chunks = []
        generated_length = 0
        while generated_length < total_frames:
            if generated_length == 0:
                chunk_start = 0
                chunk_size = min(frames_chunk, total_frames)
            else:
                chunk_start = generated_length - frames_overlap
                if chunk_start + frames_chunk > total_frames:
                    chunk_start = max(0, total_frames - frames_chunk)
                chunk_size = min(frames_chunk, total_frames - chunk_start)

            valid_size = (
                (chunk_size - 1) // self.vae_scale_factor_temporal
            ) * self.vae_scale_factor_temporal + 1
            chunk_frames = frames[:, :, chunk_start : chunk_start + valid_size].clone()
            chunk_masks = masks[:, :, chunk_start : chunk_start + valid_size]
            actual_overlap = 0
            if generated_length:
                actual_overlap = generated_length - chunk_start
                prior_frames = torch.cat(generated_chunks, dim=2)
                chunk_frames[:, :, :actual_overlap] = prior_frames[
                    :, :, chunk_start:generated_length
                ]

            chunk_latents = spatial_tiled_process(
                chunk_frames,
                chunk_masks,
                tile_num,
                tile_overlap,
                prompt_embeds,
                self.transformer,
                self.vae,
                scheduler,
                self.video_processor,
                self.vae_scale_factor_spatial,
                self.vae_scale_factor_temporal,
                self.transformer_patch_size,
                self.device,
                self.dtype,
                generator,
            )
            with torch.no_grad():
                latents_mean = torch.tensor(
                    self.vae.config.latents_mean,
                    device=self.device,
                    dtype=torch.float32,
                ).view(1, self.vae.config.z_dim, 1, 1, 1)
                latents_std = torch.tensor(
                    self.vae.config.latents_std,
                    device=self.device,
                    dtype=torch.float32,
                ).view(1, self.vae.config.z_dim, 1, 1, 1)
                chunk_latents = (chunk_latents.float() * latents_std + latents_mean).to(
                    self.vae.dtype
                )
                decoded = self.vae.decode(chunk_latents, return_dict=False)[0]
                decoded = (decoded / 2 + 0.5).clamp(0, 1)

            new_frames = (
                decoded if not generated_length else decoded[:, :, actual_overlap:]
            )
            if new_frames.shape[2] == 0:
                raise RuntimeError(
                    "Temporal chunking made no progress; reduce frames_overlap"
                )
            generated_chunks.append(new_frames)
            generated_length += new_frames.shape[2]

        output = torch.cat(generated_chunks, dim=2)[:, :, :total_frames]
        output = output[:, :, :, :original_height, :original_width]
        return output[0].permute(1, 2, 3, 0).cpu().float()


def _validate_video_inputs(
    frames_warped: torch.Tensor,
    frames_mask: torch.Tensor,
    *,
    frames_chunk: int,
    frames_overlap: int,
    tile_num: int,
    tile_overlap: int,
    inference_steps: int,
) -> None:
    if frames_warped.ndim != 4 or frames_mask.ndim != 4:
        raise ValueError("frames_warped and frames_mask must use [T, H, W, C] layout")
    if frames_warped.shape[:3] != frames_mask.shape[:3]:
        raise ValueError("frames_warped and frames_mask must share T, H, and W")
    if not frames_warped.shape[0]:
        raise ValueError("At least one frame is required")
    if frames_chunk <= 0 or not 0 <= frames_overlap < frames_chunk:
        raise ValueError(
            "frames_overlap must be non-negative and less than frames_chunk"
        )
    if tile_num <= 0 or tile_overlap < 0:
        raise ValueError("tile_num must be positive and tile_overlap non-negative")
    if inference_steps <= 0:
        raise ValueError("inference_steps must be positive")
