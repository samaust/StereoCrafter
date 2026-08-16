import gc
import hashlib
import html
import json
import math
import re
import shutil
import tempfile
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import ftfy
import torch
import torch.nn.functional as F
from diffusers import (
    AutoencoderKLWan,
    UniPCMultistepScheduler,
    WanVACETransformer3DModel,
)
from diffusers.video_processor import VideoProcessor
from filelock import FileLock
from huggingface_hub.constants import HF_HOME
from PIL import Image
from transformers import AutoTokenizer, UMT5EncoderModel

_MODEL_PRECISIONS = ("fp8", "int8", "w4a8", "fp16", "bf16")
_QUANTIZED_PRECISIONS = frozenset(("fp8", "int8", "w4a8"))
_QUANTIZED_CACHE_VERSION = 1


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "unknown"


def _resolve_model_dtype(
    model_precision: str, dtype: torch.dtype | None
) -> torch.dtype:
    if model_precision not in _MODEL_PRECISIONS:
        choices = ", ".join(_MODEL_PRECISIONS)
        raise ValueError(
            f"Unknown wan_vace model precision {model_precision!r}; choose one of: "
            f"{choices}"
        )
    expected = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(model_precision)
    if expected is not None and dtype is not None and dtype != expected:
        raise ValueError(
            f"model_precision={model_precision!r} requires dtype={expected}"
        )
    return dtype or expected or torch.bfloat16


def _torchao_config(model_precision: str):
    try:
        from diffusers import TorchAoConfig
        from torchao.quantization import (
            Float8DynamicActivationFloat8WeightConfig,
            Float8DynamicActivationInt4WeightConfig,
            Int8DynamicActivationInt8WeightConfig,
        )
    except ImportError as exc:
        raise ImportError(
            "Quantized wan_vace presets require torchao>=0.15. Install or update "
            "StereoCrafter's dependencies, or select model_precision='fp16' or "
            "'bf16'."
        ) from exc

    configs = {
        "fp8": Float8DynamicActivationFloat8WeightConfig,
        "int8": Int8DynamicActivationInt8WeightConfig,
        "w4a8": Float8DynamicActivationInt4WeightConfig,
    }
    try:
        config_factory = configs[model_precision]
    except KeyError as exc:
        raise ValueError(f"No TorchAO configuration for {model_precision!r}") from exc
    return TorchAoConfig(config_factory())


def _quantized_cache_path(
    inpainting_model: str,
    model_precision: str,
    dtype: torch.dtype,
    cache_root: str | Path | None,
) -> Path:
    root = (
        Path(cache_root) if cache_root is not None else Path(HF_HOME) / "stereocrafter"
    )
    identity = json.dumps(
        {
            "source": str(inpainting_model),
            "precision": model_precision,
            "dtype": str(dtype),
            "schema": _QUANTIZED_CACHE_VERSION,
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(identity.encode()).hexdigest()[:12]
    return root / "wan_vace" / f"{model_precision}-{digest}"


def _cache_manifest(
    inpainting_model: str, model_precision: str, dtype: torch.dtype
) -> dict[str, Any]:
    return {
        "cache_version": _QUANTIZED_CACHE_VERSION,
        "source_model": str(inpainting_model),
        "model_precision": model_precision,
        "compute_dtype": str(dtype),
        "versions": {
            "diffusers": _package_version("diffusers"),
            "torch": _package_version("torch"),
            "torchao": _package_version("torchao"),
        },
    }


def _load_quantized_transformer(
    inpainting_model: str,
    model_precision: str,
    dtype: torch.dtype,
    device: torch.device,
    *,
    cache_root: str | Path | None = None,
    rebuild: bool = False,
    model_class=WanVACETransformer3DModel,
):
    cache_path = _quantized_cache_path(
        inpainting_model, model_precision, dtype, cache_root
    )
    expected = _cache_manifest(inpainting_model, model_precision, dtype)
    manifest_path = cache_path / "stereocrafter_quantization.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    with FileLock(str(cache_path) + ".lock"):
        valid_cache = False
        if manifest_path.is_file() and not rebuild:
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                valid_cache = all(
                    manifest.get(key) == expected[key]
                    for key in (
                        "cache_version",
                        "source_model",
                        "model_precision",
                        "compute_dtype",
                    )
                )
            except (OSError, ValueError):
                valid_cache = False

        if not valid_cache:
            if cache_path.exists():
                if not rebuild:
                    raise ValueError(
                        f"Quantized model cache at {cache_path} is incompatible or "
                        "incomplete; set rebuild_quantized_cache=True to replace it"
                    )
                shutil.rmtree(cache_path)

            temporary_path = Path(
                tempfile.mkdtemp(prefix=f".{cache_path.name}-", dir=cache_path.parent)
            )
            try:
                print(
                    f"Quantizing {inpainting_model} as {model_precision} on CPU. "
                    "This one-time step can take several minutes...",
                    flush=True,
                )
                transformer = model_class.from_pretrained(
                    inpainting_model,
                    torch_dtype=dtype,
                    quantization_config=_torchao_config(model_precision),
                )
                transformer.save_pretrained(
                    temporary_path,
                    safe_serialization=False,
                )
                (temporary_path / "stereocrafter_quantization.json").write_text(
                    json.dumps(expected, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                temporary_path.replace(cache_path)
            except ImportError as exc:
                shutil.rmtree(temporary_path, ignore_errors=True)
                if model_precision == "w4a8" and "mslk" in str(exc).lower():
                    raise ImportError(
                        "The w4a8 preset requires an MSLK build compatible with "
                        "your installed PyTorch and CUDA versions."
                    ) from exc
                raise
            except Exception:
                shutil.rmtree(temporary_path, ignore_errors=True)
                raise

        print(
            f"Loading cached {model_precision} transformer from {cache_path}...",
            flush=True,
        )
        transformer = model_class.from_pretrained(
            cache_path,
            torch_dtype=dtype,
            use_safetensors=False,
        )
        print(f"Moving {model_precision} transformer to {device}...", flush=True)
        transformer = transformer.to(device)
        print(f"{model_precision} transformer is ready on {device}.", flush=True)
    return transformer


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


def build_wan_scheduler(
    scheduler: str = "euler",
    *,
    inference_steps: int,
    flow_shift: float = 5.0,
    solver_order: int = 2,
    solver_type: str = "bh2",
    lower_order_final: bool = True,
    device: str | torch.device | None = None,
):
    """Validate and construct a fresh Wan VACE inference scheduler."""
    if scheduler not in ("euler", "unipc"):
        raise ValueError(
            f"Unknown Wan VACE scheduler {scheduler!r}; choose one of: euler, unipc"
        )
    if inference_steps <= 0:
        raise ValueError("inference_steps must be positive")
    if flow_shift <= 0:
        raise ValueError("flow_shift must be positive")
    if solver_order <= 0:
        raise ValueError("solver_order must be positive")
    if solver_type not in ("bh1", "bh2"):
        raise ValueError("solver_type must be one of: bh1, bh2")

    if scheduler == "euler":
        instance = FlowMatchScheduler()
        instance.set_timesteps(
            num_inference_steps=inference_steps,
            denoising_strength=1.0,
            shift=flow_shift,
        )
        return instance

    instance = UniPCMultistepScheduler(
        num_train_timesteps=1000,
        solver_order=solver_order,
        prediction_type="flow_prediction",
        thresholding=False,
        predict_x0=True,
        solver_type=solver_type,
        lower_order_final=lower_order_final,
        use_flow_sigmas=True,
        flow_shift=flow_shift,
        final_sigmas_type="zero",
        use_dynamic_shifting=False,
    )
    instance.set_timesteps(num_inference_steps=inference_steps, device=device)
    return instance


def _scheduler_prev_sample(step_result):
    """Normalize custom Euler and Diffusers scheduler step results."""
    if isinstance(step_result, torch.Tensor):
        return step_result
    return step_result.prev_sample


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


def _tile_slices(height, width, tile_num, tile_overlap, base):
    """Return spatial tile coordinates plus dimensions used during merging."""
    tile_size = (
        int((height + tile_overlap * (tile_num - 1)) / tile_num) // base * base,
        int((width + tile_overlap * (tile_num - 1)) / tile_num) // base * base,
    )
    tile_stride = (tile_size[0] - tile_overlap, tile_size[1] - tile_overlap)
    slices = [
        (
            min(i * tile_stride[0], height - tile_size[0]),
            min(j * tile_stride[1], width - tile_size[1]),
        )
        for i in range(tile_num)
        for j in range(tile_num)
    ]
    return slices, tile_size, tile_stride


def encode_condition_tiles(
    cond_frames,
    mask_frames,
    tile_num,
    tile_overlap,
    vae,
    videoprocessor,
    vae_scale_factor_spatial,
    vae_scale_factor_temporal,
    transformer_patch_size,
    vae_device,
    dtype,
):
    """Encode every spatial tile and stage compact conditioning on CPU."""
    height, width = cond_frames.shape[-2:]
    base = vae_scale_factor_spatial * transformer_patch_size
    slices, tile_size, tile_stride = _tile_slices(
        height, width, tile_num, tile_overlap, base
    )
    records = []
    for index, (h_start, w_start) in enumerate(slices, start=1):
        print(f"  encoding tile {index}/{len(slices)}", flush=True)
        cond_tile = cond_frames[
            :, :, :, h_start : h_start + tile_size[0], w_start : w_start + tile_size[1]
        ]
        mask_tile = mask_frames[
            :, :, :, h_start : h_start + tile_size[0], w_start : w_start + tile_size[1]
        ]
        with torch.no_grad():
            condition_video, mask, reference_images = preprocess_conditions(
                video=cond_tile.permute(0, 2, 1, 3, 4),
                mask=mask_tile.permute(0, 2, 1, 3, 4),
                reference_images=None,
                batch_size=1,
                height=tile_size[0],
                width=tile_size[1],
                num_frames=cond_tile.shape[2],
                dtype=dtype,
                device=vae_device,
                video_processor=videoprocessor,
                base=base,
            )
            conditioning = prepare_video_latents(
                condition_video, mask, reference_images, vae_device, vae
            )
            mask_latents = prepare_masks(
                mask,
                reference_images,
                transformer_patch_size,
                vae_scale_factor_temporal,
                vae_scale_factor_spatial,
            )
            control = torch.cat([conditioning, mask_latents], dim=1).to(
                device="cpu", dtype=dtype
            )
        records.append(
            {
                "control": control,
                "height": tile_size[0],
                "width": tile_size[1],
                "num_frames": cond_tile.shape[2],
            }
        )
        del condition_video, mask, reference_images, conditioning, mask_latents, control
        if hasattr(vae, "clear_cache"):
            vae.clear_cache()
    return records, tile_stride


def denoise_condition_tiles(
    records,
    prompt_embeds,
    transformer,
    scheduler_factory,
    device,
    dtype,
    generator,
    vae_scale_factor_spatial,
    vae_scale_factor_temporal,
):
    """Denoise all tiles during one transformer residency window."""
    results = []
    prompt_gpu = prompt_embeds.to(device=device, dtype=dtype)
    for index, record in enumerate(records, start=1):
        noise_scheduler = scheduler_factory()
        control = record["control"].to(device=device, dtype=dtype)
        latent_frames = (record["num_frames"] - 1) // vae_scale_factor_temporal + 1
        latents = torch.randn(
            1,
            transformer.config.in_channels,
            latent_frames,
            record["height"] // vae_scale_factor_spatial,
            record["width"] // vae_scale_factor_spatial,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        total_steps = len(noise_scheduler.timesteps)
        for step_index, timestep in enumerate(noise_scheduler.timesteps, start=1):
            print(
                f"\r  denoising tile {index}/{len(records)}, "
                f"step {step_index}/{total_steps}",
                end="",
                flush=True,
            )
            timestep_tensor = timestep.unsqueeze(0).to(device, dtype=dtype)
            with torch.no_grad():
                prediction = transformer(
                    hidden_states=latents,
                    timestep=timestep_tensor,
                    encoder_hidden_states=prompt_gpu,
                    control_hidden_states=control,
                    return_dict=False,
                )[0]
            latents = _scheduler_prev_sample(
                noise_scheduler.step(prediction, timestep, latents)
            )
        results.append(latents.to("cpu"))
        del noise_scheduler, control, latents, prediction, timestep_tensor
    print(flush=True)
    return results


def merge_tile_latents(
    tiles, tile_num, tile_overlap, tile_stride, vae_scale_factor_spatial
):
    """Blend and merge completed spatial tiles entirely in host memory."""
    cols = [tiles[i : i + tile_num] for i in range(0, len(tiles), tile_num)]
    latent_stride = (
        tile_stride[0] // vae_scale_factor_spatial,
        tile_stride[1] // vae_scale_factor_spatial,
    )
    latent_overlap = (
        tile_overlap // vae_scale_factor_spatial,
        tile_overlap // vae_scale_factor_spatial,
    )
    results_cols = []
    for i, rows in enumerate(cols):
        results_rows = []
        for j, tile in enumerate(rows):
            if i > 0 and latent_overlap[0]:
                tile = blend_v(cols[i - 1][j], tile, latent_overlap[0])
            if j > 0 and latent_overlap[1]:
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
    return torch.cat(pixels, dim=3).cpu()


def _prepare_inpaint_inputs(
    frames_warped: torch.Tensor, frames_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep full-resolution inputs in host memory until a tile is processed."""

    frames = (
        frames_warped[..., :3]
        .permute(3, 0, 1, 2)
        .unsqueeze(0)
        .to(device="cpu", dtype=torch.float32)
    )
    masks = (
        frames_mask.permute(3, 0, 1, 2)
        .unsqueeze(0)
        .to(device="cpu", dtype=torch.float32)
    )
    return frames, masks


class WanVaceInpainter:
    """Wan VACE backend using explicit VAE and transformer residency phases."""

    name = "wan_vace"

    def __init__(
        self,
        base_model: str = "Wan-AI/Wan2.1-VACE-14B-diffusers",
        inpainting_model: str = "TencentARC/StereoCrafter2",
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        model_precision: str = "fp8",
        quantized_cache_dir: str | Path | None = None,
        rebuild_quantized_cache: bool = False,
        text_encoder_device: str | torch.device | None = None,
        vae_device: str | torch.device | None = None,
        sequential_offload: bool = True,
        vae_tiling: bool = True,
        vae_tile_size: int = 256,
        vae_tile_stride: int = 192,
        tokenizer: Any | None = None,
        text_encoder: Any | None = None,
        vae: Any | None = None,
        transformer: Any | None = None,
    ) -> None:
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model_precision = model_precision
        self.dtype = _resolve_model_dtype(model_precision, dtype)
        self.sequential_offload = sequential_offload
        self.vae_device = torch.device(vae_device or self.device)
        self.text_encoder_device = torch.device(text_encoder_device or "cpu")
        initial_model_device = torch.device(
            "cpu" if sequential_offload and self.device.type == "cuda" else self.device
        )
        initial_vae_device = torch.device(
            "cpu"
            if sequential_offload and self.vae_device.type == "cuda"
            else self.vae_device
        )
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(
            base_model, subfolder="tokenizer"
        )
        self.text_encoder = text_encoder or UMT5EncoderModel.from_pretrained(
            base_model, subfolder="text_encoder", torch_dtype=self.dtype
        )
        self.text_encoder.to(self.text_encoder_device)
        self.vae = vae or AutoencoderKLWan.from_pretrained(
            base_model, subfolder="vae", torch_dtype=self.dtype
        )
        self.vae.to(initial_vae_device)
        if transformer is not None:
            self.transformer = transformer.to(initial_model_device)
        elif model_precision in _QUANTIZED_PRECISIONS:
            self.transformer = _load_quantized_transformer(
                inpainting_model,
                model_precision,
                self.dtype,
                initial_model_device,
                cache_root=quantized_cache_dir,
                rebuild=rebuild_quantized_cache,
            )
        else:
            self.transformer = WanVACETransformer3DModel.from_pretrained(
                inpainting_model, torch_dtype=self.dtype
            ).to(initial_model_device)
        for model in (self.text_encoder, self.vae, self.transformer):
            model.eval()
            model.requires_grad_(False)

        self.video_processor = VideoProcessor(
            vae_scale_factor=self.vae.config.scale_factor_spatial
        )
        self.transformer_patch_size = self.transformer.config.patch_size[1]
        self.vae_scale_factor_temporal = 2 ** sum(self.vae.temperal_downsample)
        self.vae_scale_factor_spatial = 2 ** len(self.vae.temperal_downsample)
        if vae_tile_size <= 0 or vae_tile_stride <= 0:
            raise ValueError("vae_tile_size and vae_tile_stride must be positive")
        if (
            vae_tile_size % self.vae_scale_factor_spatial
            or vae_tile_stride % self.vae_scale_factor_spatial
        ):
            raise ValueError(
                "vae_tile_size and vae_tile_stride must be multiples of the VAE "
                f"spatial compression factor ({self.vae_scale_factor_spatial})"
            )
        self.vae_tiling = vae_tiling
        if vae_tiling:
            self.vae.enable_tiling(
                tile_sample_min_height=vae_tile_size,
                tile_sample_min_width=vae_tile_size,
                tile_sample_stride_height=vae_tile_stride,
                tile_sample_stride_width=vae_tile_stride,
            )
        elif hasattr(self.vae, "disable_tiling"):
            self.vae.disable_tiling()

    def _prompt_embeds(self, prompt: str) -> torch.Tensor:
        with torch.no_grad():
            embeds, _ = encode_prompt(
                [prompt],
                do_classifier_free_guidance=False,
                max_sequence_length=226,
                device=self.text_encoder_device,
                dtype=self.dtype,
                tokenizer=self.tokenizer,
                text_encoder=self.text_encoder,
            )
        return embeds.to(device="cpu", dtype=self.dtype)

    def _load_for_phase(self, label: str, model, destination: torch.device) -> float:
        started = time.monotonic()
        print(f"{label}: moving model to {destination}...", flush=True)
        if destination.type == "cuda":
            torch.cuda.reset_peak_memory_stats(destination)
        model.to(destination)
        if destination.type == "cuda":
            torch.cuda.synchronize(destination)
        print(f"{label}: model transfer complete", flush=True)
        return started

    def _finish_phase(self, label: str, model, started: float) -> None:
        if hasattr(model, "clear_cache"):
            model.clear_cache()
        peak = 0
        active_cuda = self.device.type == "cuda" or self.vae_device.type == "cuda"
        if active_cuda:
            cuda_device = self.device if self.device.type == "cuda" else self.vae_device
            torch.cuda.synchronize(cuda_device)
            peak = torch.cuda.max_memory_allocated(cuda_device)
        if self.sequential_offload and active_cuda:
            model.to("cpu")
            torch.cuda.synchronize(cuda_device)
            gc.collect()
            torch.cuda.empty_cache()
        elapsed = time.monotonic() - started
        print(
            f"{label}: complete in {elapsed:.1f}s; CUDA peak {peak / 2**30:.2f} GiB",
            flush=True,
        )

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
        scheduler: str = "euler",
        flow_shift: float = 5.0,
        solver_order: int = 2,
        solver_type: str = "bh2",
        lower_order_final: bool = True,
        seed: int | None = 0,
        prompt: str = "",
        **_: Any,
    ) -> torch.Tensor:
        total_started = time.monotonic()
        _validate_video_inputs(
            frames_warped,
            frames_mask,
            frames_chunk=frames_chunk,
            frames_overlap=frames_overlap,
            tile_num=tile_num,
            tile_overlap=tile_overlap,
            inference_steps=inference_steps,
        )
        scheduler_options = {
            "scheduler": scheduler,
            "inference_steps": inference_steps,
            "flow_shift": flow_shift,
            "solver_order": solver_order,
            "solver_type": solver_type,
            "lower_order_final": lower_order_final,
            "device": self.device,
        }
        # Construct once up front so invalid options fail before model work.
        validated_scheduler = build_wan_scheduler(**scheduler_options)
        del validated_scheduler
        summary = (
            f"Scheduler: {scheduler}; NFE/steps: {inference_steps}; "
            f"flow shift: {flow_shift:g}"
        )
        if scheduler == "unipc":
            summary += (
                f"; solver order: {solver_order}; solver type: {solver_type}; "
                f"lower order final: {lower_order_final}"
            )
        print(summary, flush=True)
        frames, masks = _prepare_inpaint_inputs(frames_warped, frames_mask)
        if masks.shape[1] not in (1, frames.shape[1]):
            raise ValueError("frames_mask must have one channel or match RGB channels")
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

        scheduler_factory = lambda: build_wan_scheduler(**scheduler_options)
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
            print(
                f"Temporal chunk {generated_length + 1}-{generated_length + valid_size} "
                f"of {total_frames} frames",
                flush=True,
            )
            chunk_frames = frames[:, :, chunk_start : chunk_start + valid_size].clone()
            chunk_masks = masks[:, :, chunk_start : chunk_start + valid_size]
            actual_overlap = 0
            if generated_length:
                actual_overlap = generated_length - chunk_start
                prior_frames = torch.cat(generated_chunks, dim=2)
                chunk_frames[:, :, :actual_overlap] = prior_frames[
                    :, :, chunk_start:generated_length
                ]

            phase = "Phase 1/3 VAE encode"
            started = self._load_for_phase(phase, self.vae, self.vae_device)
            records, tile_stride = encode_condition_tiles(
                chunk_frames,
                chunk_masks,
                tile_num,
                tile_overlap,
                self.vae,
                self.video_processor,
                self.vae_scale_factor_spatial,
                self.vae_scale_factor_temporal,
                self.transformer_patch_size,
                self.vae_device,
                self.dtype,
            )
            self._finish_phase(phase, self.vae, started)

            phase = "Phase 2/3 transformer denoise"
            started = self._load_for_phase(phase, self.transformer, self.device)
            tile_latents = denoise_condition_tiles(
                records,
                prompt_embeds,
                self.transformer,
                scheduler_factory,
                self.device,
                self.dtype,
                generator,
                self.vae_scale_factor_spatial,
                self.vae_scale_factor_temporal,
            )
            self._finish_phase(phase, self.transformer, started)
            del records
            chunk_latents = merge_tile_latents(
                tile_latents,
                tile_num,
                tile_overlap,
                tile_stride,
                self.vae_scale_factor_spatial,
            )
            del tile_latents

            phase = "Phase 3/3 VAE decode"
            started = self._load_for_phase(phase, self.vae, self.vae_device)
            with torch.no_grad():
                latents_mean = torch.tensor(
                    self.vae.config.latents_mean,
                    device=self.vae_device,
                    dtype=torch.float32,
                ).view(1, self.vae.config.z_dim, 1, 1, 1)
                latents_std = torch.tensor(
                    self.vae.config.latents_std,
                    device=self.vae_device,
                    dtype=torch.float32,
                ).view(1, self.vae.config.z_dim, 1, 1, 1)
                decode_latents = chunk_latents.to(self.vae_device)
                decode_latents = (
                    decode_latents.float() * latents_std + latents_mean
                ).to(self.vae.dtype)
                decoded = self.vae.decode(decode_latents, return_dict=False)[0]
                new_frames = (
                    decoded if not generated_length else decoded[:, :, actual_overlap:]
                )
                new_frames = (new_frames / 2 + 0.5).clamp(0, 1).cpu()
            del chunk_latents, decode_latents, decoded, latents_mean, latents_std
            self._finish_phase(phase, self.vae, started)
            if new_frames.shape[2] == 0:
                raise RuntimeError(
                    "Temporal chunking made no progress; reduce frames_overlap"
                )
            generated_chunks.append(new_frames)
            generated_length += new_frames.shape[2]

        output = torch.cat(generated_chunks, dim=2)[:, :, :total_frames]
        output = output[:, :, :, :original_height, :original_width]
        result = output[0].permute(1, 2, 3, 0).cpu().float()
        total_elapsed = time.monotonic() - total_started
        print(f"Total processing complete in {total_elapsed:.1f}s", flush=True)
        return result


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
