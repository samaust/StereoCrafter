<div align="center">
<h2>StereoCrafter: Diffusion-based Generation of Long and High-fidelity Stereoscopic 3D from Monocular Videos</h2>

Sijie Zhao*&emsp; Wenbo Hu*&emsp; Xiaodong Cun*&emsp; Yong Zhang&dagger;&emsp;
Xiaoyu Li&dagger;&emsp; Zhe Kong&emsp; Xiangjun Gao&emsp; Muyao Niu&emsp; Ying Shan

&emsp;* equal contribution &emsp; &dagger; corresponding author

<h3>Tencent AI Lab &emsp; ARC Lab, Tencent PCG</h3>

<a href="https://arxiv.org/abs/2409.07447">Paper</a> &emsp;
<a href="https://stereocrafter.github.io/">Project page</a> &emsp;
<a href="https://huggingface.co/TencentARC/StereoCrafter2">StereoCrafter2 weights</a>
</div>

This branch packages the upstream StereoCrafter v2 inference code and retains the
original SVD implementation as a selectable backend. The default backend is
`wan_vace`; `svd` remains available for the original StereoCrafter weights.

## Installation

This package is tested with Python 3.14 and CUDA 13.0. Install directly from
Git:

```bash
pip install "stereocrafter @ git+https://github.com/samaust/StereoCrafter@v2_python_package"
```

To install from a local clone:

```bash
git clone --branch v2_python_package https://github.com/samaust/StereoCrafter.git
cd StereoCrafter
pip install .
```

DepthCrafter and Forward-Warp are installed as package dependencies; this fork
does not require Git submodules.

### Optional demo dependencies

The inference demo scripts require the `demo` extra, including `fire` and video
I/O dependencies. Install it directly from Git:

```bash
pip install "stereocrafter[demo] @ git+https://github.com/samaust/StereoCrafter@v2_python_package"
```

Or, from a local clone:

```bash
pip install ".[demo]"
```

## Model backends

| Backend | Base model | Inpainting checkpoint | First-use inpainting download |
| --- | --- | --- | --- |
| `wan_vace` (default) | `Wan-AI/Wan2.1-VACE-14B-diffusers` | `TencentARC/StereoCrafter2` | ~46.5 GB (43.3 GiB) |
| `svd` | `stabilityai/stable-video-diffusion-img2vid-xt-1-1` | `TencentARC/StereoCrafter` | ~4.5 GB (4.2 GiB) |

Model arguments accept either Hugging Face identifiers or local directories.
The default identifiers are downloaded and cached by Hugging Face libraries.
The estimates cover only the source files used by the inpainting backend; they
exclude files already in the Hugging Face cache and may change as model
repositories are updated.

### Wan/VACE model precision

The Wan/VACE backend defaults to the TorchAO-backed `fp8` preset, CUDA VAE
execution, native Wan VAE tiling, and sequential model offloading for a 24 GiB
GPU such as an RTX 4090. Each temporal chunk has three GPU phases: VAE encode,
transformer denoise, and VAE decode. The inactive model is parked in system RAM,
and conditioning, tile latents, merged latents, and decoded chunks are staged
through CPU memory at phase boundaries.

| Precision | TorchAO configuration | Notes |
| --- | --- | --- |
| `fp8` (default) | FP8 dynamic activations and FP8 weights | Recommended starting point for RTX 4090-class GPUs |
| `int8` | INT8 dynamic activations and INT8 weights | Alternative 8-bit quality/performance tradeoff |
| `w4a8` | FP8 dynamic activations and grouped INT4 weights | Smallest preset; requires a compatible TorchAO MSLK kernel build |
| `fp16` | Unquantized FP16 | Compatibility option; does not fit the 34.7 GB transformer in 24 GiB |
| `bf16` | Unquantized BF16 | Original backend behavior; same storage per parameter as FP16 |

On the first quantized run, StereoCrafter downloads the original transformer,
quantizes it on CPU to avoid a temporary CUDA memory spike, and saves the result
as a reusable Diffusers model directory. This requires substantial system RAM,
additional disk space, and more startup time. Later runs load the quantized

Plan for substantial system RAM (125 GiB is the validated target) to hold the
quantized transformer and staging tensors. Model transfers add noticeable
latency, especially once per temporal chunk, but prevent the VAE and transformer
from occupying VRAM together. Phase logs report transfer completion, elapsed
time, and peak CUDA allocation.

The defaults `--vae_tile_size 256 --vae_tile_stride 192` are conservative for
4K input. Both values must be positive multiples of the VAE spatial compression
factor. Smaller tiles or strides reduce VAE peak memory at the cost of more
work; `--vae_tiling false` can be faster for small inputs or high-memory GPUs.
Explicit `--vae_device cpu` remains supported. On GPUs large enough for both
models, use `--sequential_offload false` to keep components resident and avoid
transfer overhead. CPU-only inference uses the same path with swapping as a
The provided 4K/24 GB runner also uses `--frames_chunk 41 --tile_num 3` to
leave headroom for FP8 activation buffers; larger temporal chunks or spatial
tiles may OOM even when model residency is correctly phased.
no-op.
cache directly.

By default, generated models are stored under the Hugging Face cache in
`stereocrafter/wan_vace`. Set `--quantized_cache_dir /path/to/cache` to use
another cache root. Set `--rebuild_quantized_cache true` after replacing or
updating a source model at the same identifier or local path.

Compatible alternative StereoCrafter2 transformer checkpoints can be selected
with `--inpainting_model`. They must use the Wan VACE transformer architecture
and retain the StereoCrafter inpainting fine-tune.

The depth-splatting demo is a shared prerequisite for either backend. Its first
run downloads about 4.5 GB: ~1.46 GB of SVD image-encoder/VAE files plus ~3.05
GB for `tencent/DepthCrafter`. Plan disk space accordingly if running the full
depth-splatting and inpainting workflow.

## Inference demos

Generate the depth-splatting input:

```bash
sh run_depth_splatting_inference.sh
```

Run StereoCrafter2 with the default Wan/VACE backend:

```bash
sh run_inpainting_inference.sh
```

The equivalent command is:

```bash
python demo/scripts/inpainting_inference.py \
  --backend wan_vace \
  --base_model Wan-AI/Wan2.1-VACE-14B-diffusers \
  --inpainting_model TencentARC/StereoCrafter2 \
  --model_precision fp8 \
  --input_video_path ./outputs/camel_splatting_results.mp4 \
  --save_dir ./outputs \
  --tile_num 2
```

Select the original implementation with `--backend svd`, the SVD base model,
and `TencentARC/StereoCrafter`. Both backends produce side-by-side and anaglyph
videos. The input is the four-panel video generated by the depth-splatting step.

## Python API

```python
from stereocrafter.inference.inpainting import (
    available_inpainting_backends,
    create_inpainter,
)

print(available_inpainting_backends())
inpainter = create_inpainter(
    "wan_vace",
    model_precision="fp8",
    # quantized_cache_dir="/path/to/cache",  # optional cache-root override
)
right_frames = inpainter.inpaint(warped_frames, mask_frames)
```

Use the original SVD backend explicitly:

```python
inpainter = create_inpainter(
    "svd",
    base_model="stabilityai/stable-video-diffusion-img2vid-xt-1-1",
    inpainting_model="TencentARC/StereoCrafter",
)
right_frames = inpainter.inpaint(
    warped_frames,
    mask_frames,
    frames_chunk=25,
    overlap=3,
    tile_num=1,
    num_inference_steps=8,
)
```

### Migrating from the python_package branch

The old façade functions have been removed:

```python
# Before
from stereocrafter.inference.inpainting import load_models, tiled_inpaint
pipeline = load_models()
right_frames = tiled_inpaint(warped_frames, mask_frames, pipeline)
```

Migrate to an explicit backend object:

```python
# After
from stereocrafter.inference.inpainting import create_inpainter
inpainter = create_inpainter("svd")
right_frames = inpainter.inpaint(warped_frames, mask_frames)
```

### Adding another backend

A backend factory must return an object with a string `name` and an
`inpaint(frames_warped, frames_mask, **options)` method. Frames use
`[T, H, W, C]` layout and the returned RGB tensor must be float data in
`[0, 1]`.

```python
from stereocrafter.inference.inpainting import register_inpainting_backend

register_inpainting_backend("my_backend", MyInpainter)
```

Applications can then call `create_inpainter("my_backend", ...)`.

## Outputs

The v2 demo generates side-by-side and red/cyan anaglyph videos:

<img src="assets/camel_sbs.png" alt="StereoCrafter2 side-by-side output" width="800"/>

<img src="assets/camel_anaglyph.png" alt="StereoCrafter2 anaglyph output" width="400"/>

## Acknowledgements

StereoCrafter builds on Stable Video Diffusion, DepthCrafter, Forward-Warp, and
Wan2.1-VACE.

## Citation

```bibtex
@article{zhao2024stereocrafter,
  title={Stereocrafter: Diffusion-based generation of long and high-fidelity stereoscopic 3d from monocular videos},
  author={Zhao, Sijie and Hu, Wenbo and Cun, Xiaodong and Zhang, Yong and Li, Xiaoyu and Kong, Zhe and Gao, Xiangjun and Niu, Muyao and Shan, Ying},
  journal={arXiv preprint arXiv:2409.07447},
  year={2024}
}
```
