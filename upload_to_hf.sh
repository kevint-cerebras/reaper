#!/bin/bash
set -e

# Configuration
HF_USERNAME="kevint00"
ARTIFACTS_DIR="/workspace/reaper/artifacts/Qwen3-Next-80B-A3B-Instruct/evol-codealpaca-v1/all"

echo "=================================================="
echo "Uploading Qwen3-Next Pruned Models to HuggingFace"
echo "=================================================="
echo ""

# Install/upgrade huggingface_hub
echo "Installing huggingface_hub..."
pip install -q -U huggingface_hub
echo "✅ huggingface_hub installed"
echo ""

# Login to HF
echo "🔐 Please login to HuggingFace Hub"
echo "You'll need a token with WRITE access"
echo "Get one at: https://huggingface.co/settings/tokens"
echo ""
huggingface-cli login
echo ""

# Check if artifacts exist
if [ ! -d "$ARTIFACTS_DIR" ]; then
    echo "❌ ERROR: Artifacts directory not found: $ARTIFACTS_DIR"
    exit 1
fi

echo "📦 Found artifacts directory"
echo ""

# Upload each pruned model
declare -A RATIOS=(
    ["0.5"]="50pct"
    ["0.3"]="30pct"
    ["0.1"]="10pct"
    ["0.05"]="5pct"
    ["0.02"]="2pct"
)

for ratio in 0.5 0.3 0.1 0.05 0.02; do
    MODEL_DIR="${ARTIFACTS_DIR}/${ratio}_reap_saliency"
    REPO_NAME="${HF_USERNAME}/Qwen3-Next-80B-A3B-Instruct-REAP-${RATIOS[$ratio]}"
    
    if [ ! -d "$MODEL_DIR" ]; then
        echo "⚠️  WARNING: Model directory not found, skipping: $MODEL_DIR"
        continue
    fi
    
    echo "=================================================="
    echo "📤 Uploading ${RATIOS[$ratio]} model (${ratio} compression)"
    echo "Repository: $REPO_NAME"
    echo "Source: $MODEL_DIR"
    echo ""
    
    # Check size
    SIZE=$(du -sh "$MODEL_DIR" | cut -f1)
    echo "Size: $SIZE"
    echo ""
    
    # Upload
    huggingface-cli upload \
        "$REPO_NAME" \
        "$MODEL_DIR" \
        --repo-type model \
        --create-pr false \
        --private || {
            echo "❌ Failed to upload ${RATIOS[$ratio]} model"
            continue
        }
    
    echo "✅ Successfully uploaded ${RATIOS[$ratio]} model!"
    echo ""
done

# Upload observation file if it exists
OBS_FILE=$(ls "$ARTIFACTS_DIR"/observations_*.pt 2>/dev/null | head -n1)
if [ -f "$OBS_FILE" ]; then
    echo "=================================================="
    echo "📤 Uploading observation file (for future reuse)"
    OBS_SIZE=$(du -sh "$OBS_FILE" | cut -f1)
    echo "File: $(basename $OBS_FILE)"
    echo "Size: $OBS_SIZE"
    echo ""
    
    # Create a dataset repo for the observation data
    OBS_REPO="${HF_USERNAME}/qwen3-next-80b-reap-observations"
    
    huggingface-cli upload \
        "$OBS_REPO" \
        "$OBS_FILE" \
        --repo-type dataset \
        --create-pr false \
        --private || {
            echo "⚠️  Failed to upload observation file (non-critical)"
        }
    
    echo "✅ Observation file uploaded!"
    echo ""
fi

echo "=================================================="
echo "🎉 Upload Complete!"
echo "=================================================="
echo ""
echo "Your models are now available at:"
for ratio in 0.5 0.3 0.1 0.05 0.02; do
    echo "  • https://huggingface.co/${HF_USERNAME}/Qwen3-Next-80B-A3B-Instruct-REAP-${RATIOS[$ratio]}"
done
echo ""
echo "You can now safely shut down your RunPod instance! 🚀"

