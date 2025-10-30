#!/bin/bash
set -euo pipefail
trap 'echo "❌ Error on line $LINENO. Check logs."; exit 1' ERR

# ============================================================================
# REAP Pruning Pipeline for Qwen3-Next-80B-A3B-Instruct
# ============================================================================
# This script:
# 1. Clones REAP repo
# 2. Configures Qwen3-Next support
# 3. Downloads model
# 4. Runs observation (once)
# 5. Prunes to 5 compression ratios
# ============================================================================

log() { echo "[$(date +'%H:%M:%S')] $*" | tee -a /workspace/reap_run.log; }
log_section() { echo ""; log "========== $* =========="; }

# Configuration
WORKSPACE="/workspace"
COMPRESSIONS="0.5 0.7 0.9 0.95 0.98"  # 50%, 30%, 10%, 5%, 2% remaining
MODEL="Qwen/Qwen3-Next-80B-A3B-Instruct"
DATASET="nickrosh/Evol-Instruct-Code-80k-v1"
NUM_SAMPLES=1024
MAX_LENGTH=1024  # REDUCED from 2048 to save memory
SEED=42
SALIENCY="reap"

log_section "REAP Pruning Pipeline Starting"

# Clone REAP if needed
if [ ! -d "reaper" ]; then
    log_section "Cloning REAP repository"
    git clone --depth 1 --no-recurse-submodules https://github.com/CerebrasResearch/reap.git reaper
    cd reaper
else
    log "Using existing reaper directory"
    cd reaper
fi

# Clear Python cache to avoid stale imports
log_section "Clearing Python cache"
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete

# Set environment variables
log_section "Setting environment variables"
export PYTHONPATH="/workspace/reaper/src:${PYTHONPATH:-}"
export HF_HOME="/workspace/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export PYTHONUNBUFFERED=1
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # Memory optimization
log "Environment configured"

# Install hf-transfer if not present
if ! python -c "import hf_transfer" &> /dev/null; then
    log "Installing hf-transfer for fast downloads"
    pip install -q hf-transfer
fi

# Configure Qwen models in REAP
log_section "Configuring Qwen model support"
python << 'EOF'
import sys
sys.path.insert(0, '/workspace/reaper/src')

# 1. Add Qwen2Moe to model_util.py (for test model)
model_util_path = '/workspace/reaper/src/reap/model_util.py'
with open(model_util_path, 'r') as f:
    content = f.read()

if '"Qwen2MoeForCausalLM"' not in content:
    qwen2_config = '''
    "Qwen2MoeForCausalLM": {
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
    content = content.replace('MODEL_ATTRS = {', 'MODEL_ATTRS = {' + qwen2_config)
    with open(model_util_path, 'w') as f:
        f.write(content)
    print("✓ Added Qwen2MoeForCausalLM to model_util.py")

# 2. Add Qwen3NextForCausalLM to model_util.py (for production model)
if '"Qwen3NextForCausalLM"' not in content:
    qwen3next_config = '''
    "Qwen3NextForCausalLM": {
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
    content = content.replace('MODEL_ATTRS = {', 'MODEL_ATTRS = {' + qwen3next_config)
    with open(model_util_path, 'w') as f:
        f.write(content)
    print("✓ Added Qwen3NextForCausalLM to model_util.py")

# 3. Configure observer.py for both models
observer_path = '/workspace/reaper/src/reap/observer.py'
with open(observer_path, 'r') as f:
    obs_content = f.read()

# Add Qwen3NextMoEObserverHookConfig if not present
if 'class Qwen3NextMoEObserverHookConfig' not in obs_content:
    config_class = '''
@dataclass
class Qwen3NextMoEObserverHookConfig(MoETransformerObserverConfig):
    """Observer config for Qwen3-Next models (Qwen3NextSparseMoeBlock)"""
    module_class_name_to_hook_regex: Optional[str] = "Qwen3NextSparseMoeBlock"

'''
    # Insert after Qwen3MoEObserverHookConfig
    obs_content = obs_content.replace(
        '@dataclass\nclass DeepseekV3MoEObserverHookConfig',
        config_class + '@dataclass\nclass DeepseekV3MoEObserverHookConfig'
    )
    with open(observer_path, 'w') as f:
        f.write(obs_content)
    print("✓ Added Qwen3NextMoEObserverHookConfig to observer.py")

# Reload to add registrations
with open(observer_path, 'r') as f:
    obs_content = f.read()

# Register Qwen2MoeForCausalLM
if '"Qwen2MoeForCausalLM"' not in obs_content:
    obs_content = obs_content.replace(
        'OBSERVER_CONFIG_REGISTRY = {',
        'OBSERVER_CONFIG_REGISTRY = {\n    "Qwen2MoeForCausalLM": Qwen3MoEObserverHookConfig,'
    )
    with open(observer_path, 'w') as f:
        f.write(obs_content)
    print("✓ Registered Qwen2MoeForCausalLM in observer registry")

# Register Qwen3NextForCausalLM
if '"Qwen3NextForCausalLM": Qwen3NextMoEObserverHookConfig' not in obs_content:
    obs_content = obs_content.replace(
        'OBSERVER_CONFIG_REGISTRY = {',
        'OBSERVER_CONFIG_REGISTRY = {\n    "Qwen3NextForCausalLM": Qwen3NextMoEObserverHookConfig,'
    )
    with open(observer_path, 'w') as f:
        f.write(obs_content)
    print("✓ Registered Qwen3NextForCausalLM in observer registry")

print("✅ All Qwen model configurations complete")
EOF

# Download model
log_section "Downloading model (may take 2-3 hours for 160GB)"
python << EOF
import os
from huggingface_hub import snapshot_download

cache_dir = os.environ.get("HF_HOME", "/workspace/.cache/huggingface")
print(f"Downloading to: {cache_dir}")

snapshot_download(
    "$MODEL",
    cache_dir=cache_dir,
    ignore_patterns=["*.msgpack", "*.h5"],
    resume_download=True
)
print("✅ Model downloaded successfully")
EOF

# Run observation phase (only once)
log_section "Running observation phase (~8-12 hours)"
log "Processing $NUM_SAMPLES samples at max_length=$MAX_LENGTH"

python src/reap/prune.py \
    --model-name "$MODEL" \
    --dataset-name "$DATASET" \
    --num-samples "$NUM_SAMPLES" \
    --max-length "$MAX_LENGTH" \
    --seed "$SEED" \
    --saliency "$SALIENCY" \
    --run_observer_only true \
    --output-dir artifacts 2>&1 | tee -a /workspace/reap_run.log

log "✅ Observation phase complete"

# Run pruning for each compression ratio
for COMP in $COMPRESSIONS; do
    REMAINING=$(python -c "print(int((1-$COMP)*100))")
    log_section "Pruning to ${REMAINING}% experts (compression=$COMP)"
    
    python src/reap/prune.py \
        --model-name "$MODEL" \
        --dataset-name "$DATASET" \
        --num-samples "$NUM_SAMPLES" \
        --max-length "$MAX_LENGTH" \
        --seed "$SEED" \
        --saliency "$SALIENCY" \
        --compression-ratio "$COMP" \
        --output-dir artifacts 2>&1 | tee -a /workspace/reap_run.log
    
    log "✅ Pruned model saved (${REMAINING}% experts)"
done

log_section "PIPELINE COMPLETE!"
log "Pruned models saved to: artifacts/$MODEL/evol-codealpaca-v1/pruned_models/"
log "Available variants:"
for COMP in $COMPRESSIONS; do
    REMAINING=$(python -c "print(int((1-$COMP)*100))")
    log "  - reap-seed_42-$COMP/ (${REMAINING}% experts remaining)"
done

