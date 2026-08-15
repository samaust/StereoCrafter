# Download models from Hugging Face and execute depth splatting inference.
mkdir -p ./outputs

python demo/scripts/depth_splatting_inference.py \
    --pre_trained_path stabilityai/stable-video-diffusion-img2vid-xt-1-1 \
    --unet_path tencent/DepthCrafter \
    --input_video_path ./source_video/camel.mp4 \
    --output_video_path ./outputs/camel_splatting_results.mp4
