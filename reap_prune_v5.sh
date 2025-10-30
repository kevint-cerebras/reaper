#!/bin/bash
# reap_prune_v5.sh - Production, minimal dependencies

set -euo pipefail

MODEL="Qwen/Qwen3-Next-80B-A3B-Instruct"
DATASET="theblackcat102/evol-codealpaca-v1"
COMPRESSIONS="0.5 0.7 0.9 0.95 0.98"
SEED=42
SAMPLES=1024

WORKSPACE="/workspace"
REAP_DIR="${WORKSPACE}/reaper"
LOG_FILE="${WORKSPACE}/reap_run.log"

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "REAP Pruning - V5 (MINIMAL)"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
log "✓ $GPU_COUNT GPUs"

AVAILABLE_GB=$(df -BG "$WORKSPACE" | tail -1 | awk '{print $4}' | sed 's/G//')
log "✓ ${AVAILABLE_GB}GB disk"

RAM_GB=$(free -g | awk '/^Mem:/{print $2}')
log "✓ ${RAM_GB}GB RAM"

export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_HOME="${WORKSPACE}/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export PYTHONUNBUFFERED=1

cd "$WORKSPACE"

if [ ! -d "reaper" ]; then
    log "Cloning REAP..."
    git clone --depth 1 --no-recurse-submodules https://github.com/CerebrasResearch/reap.git reaper
fi

cd "$REAP_DIR"

if [ ! -d ".venv" ]; then
    log "Installing dependencies..."
    
    if ! command -v uv &> /dev/null; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
    fi
    export PATH="/root/.local/bin:$PATH"
    
    uv venv
    source .venv/bin/activate
    
    log "Installing PyTorch..."
    uv pip install torch --index-url https://download.pytorch.org/whl/cu124 >> "$LOG_FILE" 2>&1
    
    log "Installing core dependencies..."
    uv pip install transformers accelerate datasets pyyaml >> "$LOG_FILE" 2>&1
    
    log "Installing REAP..."
    uv pip install -e . >> "$LOG_FILE" 2>&1
    
    log "✓ Setup complete"
else
    source .venv/bin/activate
fi

log "Configuring Qwen3-Next..."
python << 'EOF' >> "$LOG_FILE" 2>&1
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
EOF
log "✓ Configured"

MODEL_CACHE="${HF_HOME}/hub/models--$(echo $MODEL | sed 's|/|--|g')"
if [ -d "$MODEL_CACHE" ]; then
    log "✓ Model cached"
else
    log "Downloading model (~160GB, 2-3 hours)..."
    python << EOF >> "$LOG_FILE" 2>&1
from huggingface_hub import snapshot_download
snapshot_download("$MODEL", cache_dir="$HF_HOME", ignore_patterns=["*.msgpack", "*.h5"])
EOF
    log "✓ Downloaded"
fi

SHORT_MODEL=$(echo $MODEL | cut -d'/' -f2)
SHORT_DATASET=$(echo $DATASET | cut -d'/' -f2)
OBS_FILE="observations_${SAMPLES}_cosine-seed_${SEED}.pt"
OBS_PATH="artifacts/${SHORT_MODEL}/${SHORT_DATASET}/all/${OBS_FILE}"

if [ -f "$OBS_PATH" ]; then
    log "✓ Observations exist"
else
    log "Observation phase (8-12 hours)..."
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
        --output_file_name "$OBS_FILE" \
        >> "$LOG_FILE" 2>&1
    
    DURATION=$(($(date +%s) - START))
    log "✓ Observation: $((DURATION/3600))h $((DURATION%3600/60))m"
fi

log "Pruning 5 variants..."

for COMP in $COMPRESSIONS; do
    KEEP_PCT=$(python -c "print(int((1 - $COMP) * 100))")
    PRUNED_DIR="artifacts/${SHORT_MODEL}/${SHORT_DATASET}/pruned_models/reap-seed_${SEED}-${COMP}"
    
    if [ -d "$PRUNED_DIR" ] && [ -n "$(find "$PRUNED_DIR" -name "*.safetensors" 2>/dev/null)" ]; then
        log "✓ ${KEEP_PCT}% done"
        continue
    fi
    
    log "Pruning ${KEEP_PCT}%..."
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
        --output_file_name "$OBS_FILE" \
        >> "$LOG_FILE" 2>&1
    
    DURATION=$(($(date +%s) - START))
    log "✓ ${KEEP_PCT}%: $((DURATION/60))m"
done

log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "✓ COMPLETE!"
log "  artifacts/${SHORT_MODEL}/${SHORT_DATASET}/pruned_models/"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

