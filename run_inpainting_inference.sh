# Download models from Hugging Face automatically or replace these IDs with paths.
python demo/scripts/inpainting_inference.py \
    --backend wan_vace \
    --base_model Wan-AI/Wan2.1-VACE-14B-diffusers \
    --inpainting_model TencentARC/StereoCrafter2 \
    --input_video_path ./outputs/camel_splatting_results.mp4 \
    --save_dir ./outputs \
    --tile_num 2
