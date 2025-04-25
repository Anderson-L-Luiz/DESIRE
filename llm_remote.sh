# --- Replace with your actual cache path ---
export HUGGINGFACE_HUB_CACHE="/home/anderson/huggingface-cache" # Replace with your actual path from pwd


# --- Choose the model ID ---
#export MODEL_ID="deepseek-ai/deepseek-coder-6.7b-instruct"
export MODEL_ID="deepseek-ai/deepseek-coder-1.3b-instruct" # Smaller alternative
# export MODEL_ID="deepseek-ai/deepseek-coder-33b-instruct" # Larger alternative (needs more VRAM!)

# --- Choose a Port ---
export API_PORT=8345 # Matches the default in our backend config.py

# --- Optional: Quantization (useful for large models / limited VRAM) ---
# export QUANTIZE_ARGS="--quantize bitsandbytes-nf4" # Use 4-bit quantization
export QUANTIZE_ARGS="" # No quantization (requires more VRAM)

# --- Run the Docker container ---
docker run --gpus all \
    --shm-size 1g \
    -p ${API_PORT}:80 \
    -v ${HUGGINGFACE_HUB_CACHE}:/data \
    -e HF_HOME=/data \
    -e HF_HUB_ENABLE_HF_TRANSFER="false" \
    --hostname 0.0.0.0 \
    ghcr.io/huggingface/text-generation-inference:1.4.0 \
    --model-id ${MODEL_ID} \
    ${QUANTIZE_ARGS} \
    --max-input-length 2048 \
    --max-total-tokens 4096 \
    --max-batch-prefill-tokens 2048 \
    --trust-remote-code # IMPORTANT: DeepSeek models often require this flag!
