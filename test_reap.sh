#!/bin/bash
# test_reap.sh - Quick test with small model before running full pipeline

set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "REAP Test Suite - Small Model Validation"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

TEST_MODEL="Qwen/Qwen1.5-MoE-A2.7B"
DATASET="theblackcat102/evol-codealpaca-v1"
WORKSPACE="/workspace"

export CUDA_VISIBLE_DEVICES=0
export HF_HOME="${WORKSPACE}/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME"

# Setup
cd "$WORKSPACE"

if [ ! -d "reaper" ]; then
    echo "→ Cloning REAP repository..."
    git clone https://github.com/CerebrasResearch/reap.git reaper
fi

cd reaper

if [ ! -d ".venv" ]; then
    echo "→ Installing dependencies..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="/root/.cargo/bin:$PATH"
    bash scripts/build.sh
fi

source .venv/bin/activate

# Test 1: Observation
echo ""
echo "Test 1: Recording observations (10 samples)..."
python src/reap/prune.py \
    --model-name "$TEST_MODEL" \
    --dataset-name "$DATASET" \
    --seed 42 \
    --samples_per_category 10 \
    --run_observer_only true \
    --record_pruning_metrics_only true \
    --distance_measure cosine

if [ $? -eq 0 ]; then
    echo "✓ Observation test passed"
else
    echo "✗ Observation test failed"
    exit 1
fi

# Test 2: Pruning
echo ""
echo "Test 2: Pruning to 50%..."
python src/reap/prune.py \
    --model-name "$TEST_MODEL" \
    --dataset-name "$DATASET" \
    --compression-ratio 0.5 \
    --prune-method reap \
    --seed 42 \
    --samples_per_category 10 \
    --do-eval false \
    --smoke-test false \
    --record_pruning_metrics_only true

if [ $? -eq 0 ]; then
    echo "✓ Pruning test passed"
else
    echo "✗ Pruning test failed"
    exit 1
fi

# Test 3: Verify output
SHORT_MODEL=$(echo $TEST_MODEL | cut -d'/' -f2)
PRUNED_DIR="artifacts/${SHORT_MODEL}/evol-codealpaca-v1/pruned_models/reap-seed_42-0.5"

echo ""
echo "Test 3: Verifying output files..."

if [ -d "$PRUNED_DIR" ]; then
    echo "✓ Output directory exists"
    
    if ls $PRUNED_DIR/*.safetensors 1> /dev/null 2>&1; then
        echo "✓ Model files found"
        
        FILE_COUNT=$(ls -1 $PRUNED_DIR/*.safetensors | wc -l)
        echo "  Found $FILE_COUNT safetensors files"
    else
        echo "✗ No model files found"
        exit 1
    fi
else
    echo "✗ Output directory not created"
    exit 1
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✓ ALL TESTS PASSED"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "System is ready for Qwen3-Next-80B pruning!"
echo ""
echo "Run: ./reap_prune.sh"



