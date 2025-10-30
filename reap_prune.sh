#!/bin/bash
# reap_prune.sh - Production-ready REAP pruning for Qwen3-Next-80B-A3B-Instruct

set -euo pipefail
trap 'echo "❌ Error on line $LINENO. Check logs."; exit 1' ERR

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL="Qwen/Qwen3-Next-80B-A3B-Instruct"
DATASET="theblackcat102/evol-codealpaca-v1"
COMPRESSIONS="0.5 0.7 0.9 0.95 0.98"  # 50% 30% 10% 5% 2%
SEED=42
SAMPLES=1024

WORKSPACE="/workspace"
REAP_DIR="${WORKSPACE}/reaper/reap-source"
LOG_FILE="${WORKSPACE}/reap_run.log"

# ============================================================================
# LOGGING
# ============================================================================

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

log_error() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] ❌ ERROR: $*" | tee -a "$LOG_FILE"
}

log_success() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] ✓ $*" | tee -a "$LOG_FILE"
}

# ============================================================================
# PRE-FLIGHT CHECKS
# ============================================================================

log "Starting REAP pruning pipeline..."
log "Model: $MODEL"
log "Dataset: $DATASET"
log "Compressions: $COMPRESSIONS"

# Check GPUs
if ! nvidia-smi &> /dev/null; then
    log_error "No NVIDIA GPUs found"
    exit 1
fi

GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
log_success "Found $GPU_COUNT GPUs"

# Check disk space (need ~500GB)
AVAILABLE_GB=$(df -BG "$WORKSPACE" | tail -1 | awk '{print $4}' | sed 's/G//')
if [ "$AVAILABLE_GB" -lt 400 ]; then
    log_error "Insufficient disk space: ${AVAILABLE_GB}GB (need 400GB+)"
    exit 1
fi
log_success "Disk space: ${AVAILABLE_GB}GB available"

# Check RAM (need ~200GB)
RAM_GB=$(free -g | awk '/^Mem:/{print $2}')
if [ "$RAM_GB" -lt 150 ]; then
    log_error "Insufficient RAM: ${RAM_GB}GB (need 150GB+)"
    exit 1
fi
log_success "RAM: ${RAM_GB}GB available"

# Check internet
if ! curl -s --connect-timeout 5 https://huggingface.co > /dev/null; then
    log_error "No internet connection"
    exit 1
fi
log_success "Internet connection verified"

# Set environment
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_HOME="${WORKSPACE}/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export PYTHONUNBUFFERED=1

mkdir -p "$HF_HOME"
log_success "Environment configured"

# ============================================================================
# SETUP REAP
# ============================================================================

cd "$WORKSPACE"

# Clone repository
if [ ! -d "reaper" ]; then
    log "Cloning REAP repository..."
    if ! git clone https://github.com/cerebras/reaper.git; then
        log_error "Failed to clone repository"
        exit 1
    fi
    log_success "Repository cloned"
else
    log_success "Repository already exists"
fi

cd "$REAP_DIR"

# Install uv if needed
if ! command -v uv &> /dev/null; then
    log "Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="/root/.cargo/bin:$PATH"
    log_success "uv installed"
else
    export PATH="/root/.cargo/bin:$PATH"
    log_success "uv already installed"
fi

# Build virtual environment
if [ ! -d ".venv" ]; then
    log "Building virtual environment (this may take 10-15 minutes)..."
    if ! bash scripts/build.sh >> "$LOG_FILE" 2>&1; then
        log_error "Failed to build environment"
        exit 1
    fi
    log_success "Virtual environment built"
else
    log_success "Virtual environment exists"
fi

# Activate environment
source .venv/bin/activate
log_success "Environment activated"

# Verify Python packages
if ! python -c "import transformers, torch" 2>/dev/null; then
    log_error "Required Python packages not found"
    exit 1
fi
log_success "Python packages verified"

# ============================================================================
# CONFIGURE QWEN3-NEXT SUPPORT
# ============================================================================

log "Configuring Qwen3-Next support..."

python << 'EOF' || { log_error "Failed to configure model support"; exit 1; }
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
    print(f"✗ Failed to configure model_util.py: {e}")
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
    print(f"✗ Failed to configure observer.py: {e}")
    exit(1)
EOF

log_success "Qwen3-Next support configured"

# ============================================================================
# DOWNLOAD MODEL
# ============================================================================

log "Checking model cache..."
MODEL_CACHE="${HF_HOME}/hub/models--$(echo $MODEL | sed 's|/|--|g')"

if [ -d "$MODEL_CACHE" ]; then
    log_success "Model already cached"
else
    log "Downloading model (this will take 2-3 hours for ~160GB)..."
    
    python << EOF || { log_error "Model download failed"; exit 1; }
from huggingface_hub import snapshot_download
import os

try:
    snapshot_download(
        "$MODEL",
        cache_dir=os.environ.get("HF_HOME"),
        ignore_patterns=["*.msgpack", "*.h5", "*.ot"]
    )
    print("✓ Model downloaded")
except Exception as e:
    print(f"✗ Download failed: {e}")
    exit(1)
EOF
    
    log_success "Model downloaded and cached"
fi

# ============================================================================
# OBSERVATION PHASE
# ============================================================================

SHORT_MODEL=$(echo $MODEL | cut -d'/' -f2)
SHORT_DATASET=$(echo $DATASET | cut -d'/' -f2)
OBS_FILE="observations_${SAMPLES}_cosine-seed_${SEED}.pt"
OBS_DIR="artifacts/${SHORT_MODEL}/${SHORT_DATASET}/all"
OBS_PATH="${OBS_DIR}/${OBS_FILE}"

log "Checking for existing observations..."

if [ -f "$OBS_PATH" ]; then
    log_success "Found existing observations at $OBS_PATH"
else
    log "Starting observation phase (8-12 hours estimated)..."
    log "Recording expert activations across 48 layers × 512 experts = 24,576 experts"
    
    START_TIME=$(date +%s)
    
    if ! python src/reap/prune.py \
        --model-name "$MODEL" \
        --dataset-name "$DATASET" \
        --seed "$SEED" \
        --samples_per_category "$SAMPLES" \
        --profile true \
        --run_observer_only true \
        --record_pruning_metrics_only true \
        --distance_measure cosine \
        --output_file_name "$OBS_FILE" \
        >> "$LOG_FILE" 2>&1; then
        log_error "Observation phase failed. Check $LOG_FILE"
        exit 1
    fi
    
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    HOURS=$((DURATION / 3600))
    MINS=$(((DURATION % 3600) / 60))
    
    # Verify observation file exists
    if [ ! -f "$OBS_PATH" ]; then
        log_error "Observation file not created at $OBS_PATH"
        exit 1
    fi
    
    log_success "Observation complete in ${HOURS}h ${MINS}m"
fi

# ============================================================================
# PRUNING PHASE
# ============================================================================

log "Starting pruning phase for 5 compression ratios..."

PRUNED_COUNT=0
FAILED_COUNT=0

for COMP in $COMPRESSIONS; do
    KEEP_PCT=$(python -c "print(int((1 - $COMP) * 100))")
    EXPERTS=$(python -c "print(int(512 * (1 - $COMP)))")
    
    PRUNED_DIR="artifacts/${SHORT_MODEL}/${SHORT_DATASET}/pruned_models/reap-seed_${SEED}-${COMP}"
    
    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log "Pruning to ${KEEP_PCT}% (${EXPERTS} experts/layer)"
    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Check if already pruned
    if [ -d "$PRUNED_DIR" ] && [ -n "$(find "$PRUNED_DIR" -name "*.safetensors" 2>/dev/null)" ]; then
        log_success "Already pruned at $PRUNED_DIR"
        PRUNED_COUNT=$((PRUNED_COUNT + 1))
        continue
    fi
    
    # Prune
    START_TIME=$(date +%s)
    
    if python src/reap/prune.py \
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
        >> "$LOG_FILE" 2>&1; then
        
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        MINS=$((DURATION / 60))
        
        # Verify output
        if [ -d "$PRUNED_DIR" ] && [ -n "$(find "$PRUNED_DIR" -name "*.safetensors" 2>/dev/null)" ]; then
            log_success "Pruned to ${KEEP_PCT}% in ${MINS}m → $PRUNED_DIR"
            PRUNED_COUNT=$((PRUNED_COUNT + 1))
        else
            log_error "Pruning completed but no model files found"
            FAILED_COUNT=$((FAILED_COUNT + 1))
        fi
    else
        log_error "Pruning to ${KEEP_PCT}% failed"
        FAILED_COUNT=$((FAILED_COUNT + 1))
    fi
done

# ============================================================================
# FINAL SUMMARY
# ============================================================================

log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "PIPELINE COMPLETE"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log ""
log "Results:"
log "  ✓ Successfully pruned: $PRUNED_COUNT/5 variants"
if [ $FAILED_COUNT -gt 0 ]; then
    log "  ✗ Failed: $FAILED_COUNT/5 variants"
fi
log ""
log "Models saved at:"
log "  artifacts/${SHORT_MODEL}/${SHORT_DATASET}/pruned_models/"
log ""
log "Variants:"
for COMP in $COMPRESSIONS; do
    KEEP_PCT=$(python -c "print(int((1 - $COMP) * 100))")
    PRUNED_DIR="artifacts/${SHORT_MODEL}/${SHORT_DATASET}/pruned_models/reap-seed_${SEED}-${COMP}"
    
    if [ -d "$PRUNED_DIR" ]; then
        SIZE=$(du -sh "$PRUNED_DIR" 2>/dev/null | cut -f1)
        log "  ✓ ${KEEP_PCT}%: $PRUNED_DIR ($SIZE)"
    else
        log "  ✗ ${KEEP_PCT}%: Failed or not created"
    fi
done
log ""
log "Log file: $LOG_FILE"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ $FAILED_COUNT -eq 0 ]; then
    log_success "All variants completed successfully!"
    exit 0
else
    log_error "Some variants failed. Check $LOG_FILE for details"
    exit 1
fi



