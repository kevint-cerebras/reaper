#!/bin/bash
# test_reap_v8.sh - NO VENV, pre-downloads model

set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "REAP Test - V8"
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
    echo "→ Cloning REAP..."
    git clone --depth 1 --no-recurse-submodules https://github.com/CerebrasResearch/reap.git reaper
fi

cd reaper
export PYTHONPATH="${WORKSPACE}/reaper/src:$PYTHONPATH"

echo "→ Downloading test model (~5GB)..."
python << 'EOF'
from huggingface_hub import snapshot_download
import os
snapshot_download("Qwen/Qwen1.5-MoE-A2.7B", cache_dir=os.environ.get("HF_HOME"), ignore_patterns=["*.msgpack", "*.h5"])
print("✓ Downloaded")
EOF

echo "→ Configuring Qwen3-Next..."
python << 'EOF'
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

with open('src/reap/observer.py', 'r') as f:
    content = f.read()
if 'Qwen3NextForCausalLM' not in content:
    content = content.replace(
        'OBSERVER_CONFIG_REGISTRY = {',
        'OBSERVER_CONFIG_REGISTRY = {\n    "Qwen3NextForCausalLM": Qwen3MoEObserverHookConfig,'
    )
    with open('src/reap/observer.py', 'w') as f:
        f.write(content)
print("✓ Configured")
EOF

echo ""
echo "Test 1: Observation (10 samples)..."
python src/reap/prune.py \
    --model-name "$TEST_MODEL" \
    --dataset-name "$DATASET" \
    --seed 42 \
    --samples_per_category 10 \
    --run_observer_only true \
    --record_pruning_metrics_only true \
    --distance_measure cosine

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

SHORT_MODEL=$(echo $TEST_MODEL | cut -d'/' -f2)
PRUNED_DIR="artifacts/${SHORT_MODEL}/evol-codealpaca-v1/pruned_models/reap-seed_42-0.5"

echo ""
if [ -d "$PRUNED_DIR" ] && ls $PRUNED_DIR/*.safetensors 1> /dev/null 2>&1; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "✓ ALL TESTS PASSED!"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "Run: ./reap_prune_v8.sh"
else
    echo "✗ Test failed"
    exit 1
fi

