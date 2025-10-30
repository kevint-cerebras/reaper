#!/bin/bash
# run_qwen3next_pruning.sh - Final production script for Qwen3-Next-80B pruning

set -euo pipefail

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Qwen3-Next-80B REAP Pruning - Full Pipeline"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

MODEL="Qwen/Qwen3-Next-80B-A3B-Instruct"
DATASET="theblackcat102/evol-codealpaca-v1"
COMPRESSIONS="0.5 0.7 0.9 0.95 0.98"
SEED=42
SAMPLES=1024

WORKSPACE="/workspace"
LOG_FILE="${WORKSPACE}/reap_run.log"

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

# Setup environment
export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_HOME="${WORKSPACE}/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export PYTHONUNBUFFERED=1

cd "$WORKSPACE"

# Clone REAP if needed
if [ ! -d "reaper" ]; then
    log "Cloning REAP..."
    git clone --depth 1 --no-recurse-submodules https://github.com/CerebrasResearch/reap.git reaper
fi

cd "${WORKSPACE}/reaper"
export PYTHONPATH="${WORKSPACE}/reaper/src"

log "Configuring Qwen models..."
python << 'EOF'
# Configure model_util.py
with open('src/reap/model_util.py', 'r') as f:
    content = f.read()

if 'Qwen2MoeForCausalLM' not in content:
    qwen2 = '''    "Qwen2MoeForCausalLM": {
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
    content = content.replace('"Qwen3MoeForCausalLM": {', qwen2 + '"Qwen3MoeForCausalLM": {')

if 'Qwen3NextForCausalLM' not in content:
    qwen3next = '''    "Qwen3NextForCausalLM": {
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
    content = content.replace('"Qwen3MoeForCausalLM": {', qwen3next + '"Qwen3MoeForCausalLM": {')

with open('src/reap/model_util.py', 'w') as f:
    f.write(content)

# Configure observer.py
with open('src/reap/observer.py', 'r') as f:
    content = f.read()

if 'Qwen2MoeForCausalLM' not in content or 'Qwen3NextForCausalLM' not in content:
    content = content.replace(
        'OBSERVER_CONFIG_REGISTRY = {',
        'OBSERVER_CONFIG_REGISTRY = {\n    "Qwen2MoeForCausalLM": Qwen3MoEObserverHookConfig,\n    "Qwen3NextForCausalLM": Qwen3MoEObserverHookConfig,'
    )
    with open('src/reap/observer.py', 'w') as f:
        f.write(content)

print("✓ Configured")
EOF

log "✓ Models configured"

# Download model
MODEL_CACHE="${HF_HOME}/hub/models--$(echo $MODEL | sed 's|/|--|g')"
if [ -d "$MODEL_CACHE" ]; then
    log "✓ Model already cached"
else
    log "Downloading Qwen3-Next-80B (~160GB, 2-3 hours)..."
    python << EOF
from huggingface_hub import snapshot_download
snapshot_download("$MODEL", cache_dir="$HF_HOME", ignore_patterns=["*.msgpack", "*.h5"])
print("✓ Downloaded")
EOF
    log "✓ Model downloaded"
fi

# Observation phase
SHORT_MODEL=$(echo $MODEL | cut -d'/' -f2)
SHORT_DATASET=$(echo $DATASET | cut -d'/' -f2)
OBS_FILE="observations_${SAMPLES}_cosine-seed_${SEED}.pt"
OBS_PATH="artifacts/${SHORT_MODEL}/${SHORT_DATASET}/all/${OBS_FILE}"

if [ -f "$OBS_PATH" ]; then
    log "✓ Observations already exist"
else
    log "Starting observation phase (8-12 hours)..."
    START=$(date +%s)
    
    python src/reap/prune.py \
        --model-name "$MODEL" \
        --dataset-name "$DATASET" \
        --seed "$SEED" \
        --samples_per_category "$SAMPLES" \
        --profile true \
        --run_observer_only true \
        --record_pruning_metrics_only true \
        --distance_measure cosine \
        --output_file_name "$OBS_FILE"
    
    DURATION=$(($(date +%s) - START))
    log "✓ Observation complete: $((DURATION/3600))h $((DURATION%3600/60))m"
fi

# Pruning phase
log "Starting pruning for 5 compression ratios..."

for COMP in $COMPRESSIONS; do
    KEEP_PCT=$(python -c "print(int((1 - $COMP) * 100))")
    PRUNED_DIR="artifacts/${SHORT_MODEL}/${SHORT_DATASET}/pruned_models/reap-seed_${SEED}-${COMP}"
    
    if [ -d "$PRUNED_DIR" ] && [ -n "$(find "$PRUNED_DIR" -name "*.safetensors" 2>/dev/null)" ]; then
        log "✓ ${KEEP_PCT}% already done"
        continue
    fi
    
    log "Pruning to ${KEEP_PCT}%..."
    START=$(date +%s)
    
    python src/reap/prune.py \
        --model-name "$MODEL" \
        --dataset-name "$DATASET" \
        --compression-ratio "$COMP" \
        --prune-method reap \
        --seed "$SEED" \
        --samples_per_category "$SAMPLES" \
        --do-eval false \
        --smoke-test false \
        --record_pruning_metrics_only true \
        --output_file_name "$OBS_FILE"
    
    DURATION=$(($(date +%s) - START))
    log "✓ ${KEEP_PCT}% complete: $((DURATION/60))m"
done

log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "✓ ALL DONE!"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log ""
log "Pruned models saved at:"
log "  ${WORKSPACE}/reaper/artifacts/${SHORT_MODEL}/${SHORT_DATASET}/pruned_models/"
log ""
log "Models created:"
log "  - 50% (256 experts/layer)"
log "  - 30% (154 experts/layer)"
log "  - 10% (51 experts/layer)"
log "  - 5% (26 experts/layer)"
log "  - 2% (10 experts/layer)"
log ""

