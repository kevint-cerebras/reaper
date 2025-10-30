#!/bin/bash
# reap_prune_v3.sh - Simple production script, NO SUBMODULES

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
log "REAP Pruning Pipeline - Simple Version"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check GPUs
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
log "✓ Found $GPU_COUNT GPUs"

# Check disk
AVAILABLE_GB=$(df -BG "$WORKSPACE" | tail -1 | awk '{print $4}' | sed 's/G//')
log "✓ Disk: ${AVAILABLE_GB}GB available"

# Check RAM
RAM_GB=$(free -g | awk '/^Mem:/{print $2}')
log "✓ RAM: ${RAM_GB}GB"

# Set environment
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_HOME="${WORKSPACE}/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export PYTHONUNBUFFERED=1

cd "$WORKSPACE"

# Clone repo (NO SUBMODULES)
if [ ! -d "reaper" ]; then
    log "Cloning REAP repository (NO SUBMODULES)..."
    git clone --depth 1 --no-recurse-submodules https://github.com/CerebrasResearch/reap.git reaper
    log "✓ Repository cloned"
else
    log "✓ Repository exists"
fi

cd "$REAP_DIR"

# Install dependencies
if [ ! -d ".venv" ]; then
    log "Installing dependencies..."
    
    if ! command -v uv &> /dev/null; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
    fi
    export PATH="/root/.local/bin:$PATH"
    
    uv venv
    source .venv/bin/activate
    
    log "Installing PyTorch..."
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 >> "$LOG_FILE" 2>&1
    
    log "Installing transformers & dependencies..."
    uv pip install transformers==4.55.0 accelerate datasets >> "$LOG_FILE" 2>&1
    
    log "Installing REAP package..."
    uv pip install -e . >> "$LOG_FILE" 2>&1
    
    log "✓ Dependencies installed"
else
    source .venv/bin/activate
    log "✓ Environment activated"
fi

# Configure Qwen3-Next
log "Configuring Qwen3-Next support..."
python << 'EOF' >> "$LOG_FILE" 2>&1
# Configure model_util.py
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

# Configure observer.py
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
log "✓ Qwen3-Next configured"

# Download model
log "Checking model cache..."
MODEL_CACHE="${HF_HOME}/hub/models--$(echo $MODEL | sed 's|/|--|g')"

if [ -d "$MODEL_CACHE" ]; then
    log "✓ Model cached"
else
    log "Downloading model (~160GB, 2-3 hours)..."
    python << EOF >> "$LOG_FILE" 2>&1
from huggingface_hub import snapshot_download
snapshot_download("$MODEL", cache_dir="$HF_HOME", ignore_patterns=["*.msgpack", "*.h5"])
EOF
    log "✓ Model downloaded"
fi

# Observation phase
SHORT_MODEL=$(echo $MODEL | cut -d'/' -f2)
SHORT_DATASET=$(echo $DATASET | cut -d'/' -f2)
OBS_FILE="observations_${SAMPLES}_cosine-seed_${SEED}.pt"
OBS_PATH="artifacts/${SHORT_MODEL}/${SHORT_DATASET}/all/${OBS_FILE}"

if [ -f "$OBS_PATH" ]; then
    log "✓ Observations exist"
else
    log "Starting observation (8-12 hours)..."
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
    log "✓ Observation complete in $((DURATION/3600))h $((DURATION%3600/60))m"
fi

# Pruning phase
log "Starting pruning for 5 compression ratios..."

for COMP in $COMPRESSIONS; do
    KEEP_PCT=$(python -c "print(int((1 - $COMP) * 100))")
    PRUNED_DIR="artifacts/${SHORT_MODEL}/${SHORT_DATASET}/pruned_models/reap-seed_${SEED}-${COMP}"
    
    if [ -d "$PRUNED_DIR" ] && [ -n "$(find "$PRUNED_DIR" -name "*.safetensors" 2>/dev/null)" ]; then
        log "✓ Already pruned to ${KEEP_PCT}%"
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
        --output_file_name "$OBS_FILE" \
        >> "$LOG_FILE" 2>&1
    
    DURATION=$(($(date +%s) - START))
    log "✓ Pruned to ${KEEP_PCT}% in $((DURATION/60))m"
done

log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "✓ COMPLETE! Models saved at:"
log "  artifacts/${SHORT_MODEL}/${SHORT_DATASET}/pruned_models/"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

