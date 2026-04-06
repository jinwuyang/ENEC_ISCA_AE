#!/bin/bash

# Environment Setup
source /data/wja/ascend/ascend-toolkit/set_env.sh
conda activate hans

# Define list of models to test
MODELS=(
    "./models/BF16/Qwen3-32B"
    "./models/BF16/falcon-40b"
)

# Define list of Batch Sizes
BATCH_SIZES=(1 2 4 8 16 32)

# Ensure output directory exists
mkdir -p logs

# Device Configurations (Single NPU vs. Multiple NPUs)
DEVICE_CONFIGS=("0" "0,1")

for devices in "${DEVICE_CONFIGS[@]}"
do
    export ASCEND_RT_VISIBLE_DEVICES=$devices
    echo "🖥️  Current Devices in use: $devices"
    
    for model in "${MODELS[@]}"
    do
        echo "========================================================="
        echo "🔥 Starting Benchmark for Model: ${model##*/}"
        echo "========================================================="

        for bs in "${BATCH_SIZES[@]}"
        do
            echo "[$(date +%T)] 🚀 Testing: Model=${model##*/}, BatchSize=$bs"
            
            # Run Inference Script
            # Note: Overriding default values using --model and --batch_size arguments
            python python/inference.py \
                --model "$model" \
                --batch_size "$bs" \
                --seq_len 128 \
                --max_new_tokens 32
                
            if [ $? -ne 0 ]; then
                echo "❌ Error: Model $model failed at BS=$bs. Skipping current configuration."
            fi
        done
    done
done

echo "🎉 All benchmarks completed! Results summarized in: python/benchmark_results3_bs.csv"