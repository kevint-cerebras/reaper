#!/bin/bash
# test_reap_v4.sh - Simple test, NO submodules, CORRECT torch version

set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "REAP Test Suite - V4 (FIXED)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

TEST_MODEL="Qwen/Qwen1.5-MoE-A2.7B"
DATASET="theblackcat102/evol-codealpaca-v1"
WORKSPACE="/workspace"

export CUDA_VISIBLE_DEVICES=0
export HF_HOME="${WORKSPACE}/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME"

cd "$WORKSPACE"

if [ ! -d "reaper" ]; then
    echo "→ Cloning REAP repository (NO SUBMODULES)..."
    git clone --depth 1 --no-recurse-submodules https://github.com/CerebrasResearch/reap.git reaper
fi

cd reaper

if [ ! -d ".venv" ]; then
    echo "→ Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="/root/.local/bin:$PATH"
    
    echo "→ Creating virtual environment..."
    uv venv
    source .venv/bin/activate
    
    echo "→ Installing PyTorch (latest stable with CUDA 12.4)..."
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    
    echo "→ Installing transformers & core deps..."
    uv pip install transformers==4.55.0 accelerate datasets
    
    echo "→ Installing REAP package..."
    uv pip install -e .
    
    echo "✓ Dependencies installed"
else
    source .venv/bin/activate
fi

# Configure Qwen3-Next support
echo "→ Configuring Qwen3-Next support..."
python << 'EOF'
import re

# Configure model_util.py
try:
    with open('src/reap/model_util.py', 'r') as f:
        content = f.read()
    
    if 'Qwen3NextForCausalLM' not in content:
        config = '''    "Qwen3NextForCausalLM": {
        "moe_block": "mlp",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": False,
        "router": "gate",
        "num_experts": "num_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    },
'''
        content = content.replace('"Qwen3MoeForCausalLM": {', config + '"Qwen3MoeForCausalLM": {')
        with open('src/reap/model_util.py', 'w') as f:
            f.write(content)
        print("✓ model_util.py configured")
    else:
        print("✓ model_util.py already configured")
except Exception as e:
    print(f"✗ Failed: {e}")
    exit(1)

# Configure observer.py
try:
    with open('src/reap/observer.py', 'r') as f:
        content = f.read()
    
    if 'Qwen3NextForCausalLM' not in content:
        content = content.replace(
            'OBSERVER_CONFIG_REGISTRY = {',
            'OBSERVER_CONFIG_REGISTRY = {\n    "Qwen3NextForCausalLM": Qwen3MoEObserverHookConfig,'
        )
        with open('src/reap/observer.py', 'w') as f:
            f.write(content)
        print("✓ observer.py configured")
    else:
        print("✓ observer.py already configured")
except Exception as e:
    print(f"✗ Failed: {e}")
    exit(1)
EOF

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

SHORT_MODEL=$(echo $TEST_MODEL | cut -d'/' -f2)
PRUNED_DIR="artifacts/${SHORT_MODEL}/evol-codealpaca-v1/pruned_models/reap-seed_42-0.5"

echo ""
echo "Test 3: Verifying output files..."

if [ -d "$PRUNED_DIR" ] && ls $PRUNED_DIR/*.safetensors 1> /dev/null 2>&1; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "✓ ALL TESTS PASSED"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "System is ready! Run: ./reap_prune_v4.sh"
else
    echo "✗ No model files found"
    exit 1
fi

