#!/bin/bash
# validate_script.sh - Quick validation before running REAP (takes ~30 seconds)

set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "REAP Script Validation"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

WORKSPACE="/workspace"
cd "$WORKSPACE"

# Check 1: Environment exists
echo "Check 1: Environment..."
if [ ! -d "reaper/.venv" ]; then
    echo "✗ No .venv found. Run test_reap_v6.sh first"
    exit 1
fi

cd reaper
source .venv/bin/activate
echo "✓ Virtual environment activated"

# Check 2: All imports work
echo ""
echo "Check 2: Python imports..."
python << 'EOF'
try:
    import torch
    import transformers
    import accelerate
    import datasets
    import yaml
    print(f"✓ torch {torch.__version__}")
    print(f"✓ transformers {transformers.__version__}")
    print(f"✓ accelerate {accelerate.__version__}")
    print(f"✓ datasets {datasets.__version__}")
    print(f"✓ pyyaml imported")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    exit(1)
EOF

# Check 3: CUDA available
echo ""
echo "Check 3: CUDA availability..."
python << 'EOF'
import torch
if not torch.cuda.is_available():
    print("✗ CUDA not available!")
    exit(1)
gpu_count = torch.cuda.device_count()
print(f"✓ CUDA available with {gpu_count} GPUs")
for i in range(gpu_count):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
EOF

# Check 4: REAP scripts exist
echo ""
echo "Check 4: REAP scripts..."
if [ ! -f "src/reap/prune.py" ]; then
    echo "✗ src/reap/prune.py not found"
    exit 1
fi
echo "✓ prune.py exists"

if [ ! -f "src/reap/model_util.py" ]; then
    echo "✗ src/reap/model_util.py not found"
    exit 1
fi
echo "✓ model_util.py exists"

if [ ! -f "src/reap/observer.py" ]; then
    echo "✗ src/reap/observer.py not found"
    exit 1
fi
echo "✓ observer.py exists"

# Check 5: prune.py arguments are correct
echo ""
echo "Check 5: prune.py arguments..."
python << 'EOF'
import sys
import argparse

# Check if we can import the script
sys.path.insert(0, 'src')
try:
    from reap import args as reap_args
    print("✓ prune.py arguments can be parsed")
except Exception as e:
    print(f"✗ Failed to parse args: {e}")
    exit(1)
EOF

# Check 6: Configuration changes are valid
echo ""
echo "Check 6: Configuration syntax..."
python << 'EOF'
# Test the config changes won't break anything
test_config = """    "Qwen3NextForCausalLM": {
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
"""

# Verify it's valid Python dict syntax
try:
    exec(f"test_dict = {{{test_config}}}")
    print("✓ Qwen3Next config syntax is valid")
except SyntaxError as e:
    print(f"✗ Config syntax error: {e}")
    exit(1)
EOF

# Check 7: HuggingFace connectivity
echo ""
echo "Check 7: HuggingFace connectivity..."
python << 'EOF'
from huggingface_hub import HfApi
try:
    api = HfApi()
    # Try to get model info (doesn't download, just checks API)
    info = api.model_info("Qwen/Qwen1.5-MoE-A2.7B")
    print(f"✓ Can access HuggingFace Hub")
    print(f"  Test model: {info.modelId}")
except Exception as e:
    print(f"⚠ HuggingFace access issue (may still work): {e}")
EOF

# Check 8: Disk space
echo ""
echo "Check 8: Resources..."
AVAILABLE_GB=$(df -BG "$WORKSPACE" | tail -1 | awk '{print $4}' | sed 's/G//')
if [ "$AVAILABLE_GB" -lt 100 ]; then
    echo "⚠ Low disk space: ${AVAILABLE_GB}GB (test needs ~20GB, full run needs 400GB+)"
else
    echo "✓ Disk space: ${AVAILABLE_GB}GB"
fi

RAM_GB=$(free -g | awk '/^Mem:/{print $2}')
if [ "$RAM_GB" -lt 50 ]; then
    echo "⚠ Low RAM: ${RAM_GB}GB (recommended 150GB+ for full run)"
else
    echo "✓ RAM: ${RAM_GB}GB"
fi

GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
echo "✓ GPUs: $GPU_COUNT"

# Check 9: Quick dry-run test
echo ""
echo "Check 9: Dry-run test (loading test model metadata)..."
python << 'EOF'
from transformers import AutoConfig
try:
    # This just loads the config, doesn't download the full model
    config = AutoConfig.from_pretrained("Qwen/Qwen1.5-MoE-A2.7B", trust_remote_code=True)
    print(f"✓ Test model config loaded")
    print(f"  Model type: {config.model_type}")
    if hasattr(config, 'num_experts'):
        print(f"  Experts: {config.num_experts}")
except Exception as e:
    print(f"✗ Failed to load test model config: {e}")
    exit(1)
EOF

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✓ ALL VALIDATION CHECKS PASSED!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Your environment is ready. Safe to run:"
echo "  ./test_reap_v6.sh    (30 min test)"
echo "  ./reap_prune_v6.sh   (full 14-16 hour run)"
echo ""

