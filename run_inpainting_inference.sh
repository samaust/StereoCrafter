# Download models from Hugging Face automatically or replace these IDs with paths.
DIFFUSERS_ATTN_BACKEND=sage \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=. python demo/scripts/inpainting_inference.py \
    --backend wan_vace \
    --base_model Wan-AI/Wan2.1-VACE-14B-diffusers \
    --inpainting_model TencentARC/StereoCrafter2 \
    --model_precision fp8 \
    --input_video_path ./outputs/camel_splatting_results.mp4 \
    --vae_device cuda \
    --sequential_offload true \
    --vae_tiling true \
    --vae_tile_size 512 \
    --vae_tile_stride 384 \
    --save_dir ./outputs \
    --scheduler euler \
    --flow_shift 5.0 \
    --tile_num 3 \
    --frames_chunk 41
