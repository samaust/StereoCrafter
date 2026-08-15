# Download models from huggingface and execute inpainting inference
python demo/scripts/inpainting_inference.py \
    --pre_trained_path stabilityai/stable-video-diffusion-img2vid-xt-1-1 \
    --unet_path TencentARC/StereoCrafter \
    --input_video_path ./outputs/camel_splatting_results.mp4 \
    --save_dir ./outputs \
    --tile_num 2
