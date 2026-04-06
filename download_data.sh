#!/bin/bash

# Environment Setup
export HF_ENDPOINT=https://hf-mirror.com

# Model Mapping Table (ModelScope_ID : Hugging Face_ID)
declare -A MODEL_MAP
MODEL_MAP["LLM-Research/OLMo-1B-hf"]="allenai/OLMo-1B-hf"
MODEL_MAP["AI-ModelScope/bert-base-uncased"]="google-bert/bert-base-uncased"
MODEL_MAP["AI-ModelScope/wav2vec2-large-xlsr-53-english"]="jonatasgrosman/wav2vec2-large-xlsr-53-english"
MODEL_MAP["LLM-Research/CapybaraHermes-2.5-Mistral-7B"]="argilla/CapybaraHermes-2.5-Mistral-7B"
MODEL_MAP["AI-ModelScope/stable-video-diffusion-img2vid-fp16"]="stabilityai/stable-video-diffusion-img2vid"
MODEL_MAP["AI-ModelScope/falcon-7b"]="tiiuae/falcon-7b"
MODEL_MAP["AI-ModelScope/falcon-40b"]="tiiuae/falcon-40b"
MODEL_MAP["deepseek-ai/deepseek-llm-7b-base"]="deepseek-ai/deepseek-llm-7b-base"
MODEL_MAP["qwen/Qwen3-8B"]="Qwen/Qwen3-8B"
MODEL_MAP["qwen/Qwen3-32B"]="Qwen/Qwen3-32B"
# Note: ModelScope IDs must include the LLM-Research prefix
MODEL_MAP["LLM-Research/Meta-Llama-3.1-8B-Instruct"]="meta-llama/Llama-3.1-8B-Instruct"

# Precision Grouping (Directly using MODEL_MAP Keys)
FP32="LLM-Research/OLMo-1B-hf AI-ModelScope/bert-base-uncased AI-ModelScope/wav2vec2-large-xlsr-53-english"
FP16="LLM-Research/CapybaraHermes-2.5-Mistral-7B AI-ModelScope/stable-video-diffusion-img2vid-fp16"
BF16="AI-ModelScope/falcon-7b AI-ModelScope/falcon-40b deepseek-ai/deepseek-llm-7b-base qwen/Qwen3-8B qwen/Qwen3-32B LLM-Research/Meta-Llama-3.1-8B-Instruct"

download_logic() {
    local DTYPE=$1
    local MODELS=$2
    for MS_ID in $MODELS; do
        HF_ID=${MODEL_MAP[$MS_ID]}
        # Extract directory name (e.g., Qwen3-8B)
        DIR_NAME=${HF_ID##*/}
        SAVE_PATH="models/$DTYPE/$DIR_NAME"

        echo "------------------------------------------------"
        echo "⬇️ Processing: $DIR_NAME (Precision: $DTYPE)"
        
        # Attempt 1: Hugging Face
        echo "   [Attempt 1/2] Downloading from Hugging Face..."
        hf download "$HF_ID" --local-dir "$SAVE_PATH"
        
        if [ $? -eq 0 ]; then
            echo "   ✅ HF download successful"
        else
            echo "   ⚠️ HF failed (Access 403), switching to ModelScope..."
            # Attempt 2: ModelScope
            modelscope download --model "$MS_ID" --local_dir "$SAVE_PATH"
            [ $? -eq 0 ] && echo "   ✅ ModelScope download successful" || echo "   ❌ Download failed for this model"
        fi
    done
}

# Execute download logic for each precision group
download_logic "FP32" "$FP32"
download_logic "FP16" "$FP16"
download_logic "BF16" "$BF16"