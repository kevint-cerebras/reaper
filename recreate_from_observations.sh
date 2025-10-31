#!/bin/bash
set -e

echo "=========================================="
echo "Recreating Pruned Models from Observations"
echo "=========================================="
echo ""

# Configuration
MODEL="Qwen/Qwen3-Next-80B-A3B-Instruct"
DATASET="evol-codealpaca-v1"
SEED=42
MAX_LENGTH=1024
OBS_FILE="observations_1024_cosine-seed_42.pt"
OBS_PATH="/workspace/reaper/artifacts/Qwen3-Next-80B-A3B-Instruct/evol-codealpaca-v1/all/${OBS_FILE}"
COMPRESSIONS="0.5 0.3 0.1 0.05 0.02"

cd /workspace/reaper

# Step 1: Validate observation file exists
echo "Step 1: Validating observation file..."
if [ ! -f "$OBS_PATH" ]; then
    echo "❌ Observation file not found: $OBS_PATH"
    echo ""
    echo "Download it from HuggingFace:"
    echo "  python3 << 'EOF'"
    echo "  from huggingface_hub import hf_hub_download"
    echo "  import os"
    echo "  hf_hub_download("
    echo "      repo_id='kevint00/qwen3-next-80b-reap-observations',"
    echo "      filename='observations_1024_cosine-seed_42.pt',"
    echo "      local_dir='/workspace/reaper/artifacts/Qwen3-Next-80B-A3B-Instruct/evol-codealpaca-v1/all/',"
    echo "      token=os.environ.get('HF_TOKEN')"
    echo "  )"
    echo "  EOF"
    exit 1
fi

echo "✅ Found observation file ($(du -h $OBS_PATH | cut -f1))"
echo ""

# Step 2: Validate observation file can be loaded
echo "Step 2: Validating observation file integrity..."
python3 << 'EOF'
import torch
import sys

obs_path = "/workspace/reaper/artifacts/Qwen3-Next-80B-A3B-Instruct/evol-codealpaca-v1/all/observations_1024_cosine-seed_42.pt"

try:
    print(f"Loading {obs_path}...")
    data = torch.load(obs_path, map_location='cpu')
    
    print(f"✅ File loaded successfully")
    print(f"   Type: {type(data)}")
    
    if isinstance(data, dict):
        print(f"   Keys: {list(data.keys())}")
        if 'expert_data' in data:
            print(f"   Expert data shape: {len(data['expert_data'])} entries")
    
    print("\n✅ Observation file is valid!")
    sys.exit(0)
    
except Exception as e:
    print(f"\n❌ ERROR: Failed to load observation file")
    print(f"   {e}")
    sys.exit(1)
EOF

if [ $? -ne 0 ]; then
    echo "❌ Observation file validation failed"
    exit 1
fi

echo ""

# Step 3: Clean old pruned models (optional - comment out to keep)
# echo "Step 3: Cleaning old pruned models..."
# rm -rf /workspace/reaper/artifacts/Qwen3-Next-80B-A3B-Instruct/evol-codealpaca-v1/pruned_models/*
# echo "✅ Cleaned"
# echo ""

# Step 4: Run pruning for each compression ratio
echo "Step 3: Running pruning for all compression ratios..."
echo "Compressions: $COMPRESSIONS"
echo ""

export PYTHONPATH="${PYTHONPATH:-}:/workspace/reaper"

for COMP in $COMPRESSIONS; do
    echo "=========================================="
    echo "Pruning to ${COMP} compression (keeping $(echo "$COMP * 100" | bc)% of experts)"
    echo "=========================================="
    
    python src/reap/prune.py \
        --model_name "$MODEL" \
        --dataset_name "$DATASET" \
        --compression_ratio "$COMP" \
        --prune_method reap_saliency \
        --seed "$SEED" \
        --samples_per_category 1024 \
        --model_max_length "$MAX_LENGTH" \
        --do_eval false \
        --smoke_test false \
        --record_pruning_metrics_only true \
        --output_file_name "$OBS_FILE"
    
    if [ $? -eq 0 ]; then
        echo "✅ Completed ${COMP} compression"
    else
        echo "❌ Failed ${COMP} compression"
        exit 1
    fi
    
    echo ""
done

echo "=========================================="
echo "🎉 All pruned models created successfully!"
echo "=========================================="
echo ""
echo "Models saved to:"
echo "  /workspace/reaper/artifacts/Qwen3-Next-80B-A3B-Instruct/evol-codealpaca-v1/pruned_models/"
echo ""
echo "List of models:"
ls -lh /workspace/reaper/artifacts/Qwen3-Next-80B-A3B-Instruct/evol-codealpaca-v1/pruned_models/
echo ""
echo "Next steps:"
echo "  1. Test models: python3 /workspace/test_local_pruned_models.py"
echo "  2. Upload to HuggingFace if tests pass"

