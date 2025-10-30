#!/bin/bash
# fix_qwen3next_observer.sh - Fix observer configuration for Qwen3Next

set -e

cd /workspace/reaper
export PYTHONPATH="/workspace/reaper/src"

echo "Fixing Qwen3Next observer configuration..."

python << 'EOF'
import re

# Fix model_util.py
print("Configuring model_util.py...")
with open('src/reap/model_util.py', 'r') as f:
    content = f.read()

# Make sure Qwen3NextForCausalLM is configured
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
        "num_experts_per_tok": "top_k",
    },
'''
    content = content.replace('"Qwen3MoeForCausalLM": {', qwen3next + '"Qwen3MoeForCausalLM": {')
    with open('src/reap/model_util.py', 'w') as f:
        f.write(content)
    print("✓ Added Qwen3NextForCausalLM config")
else:
    print("✓ Qwen3NextForCausalLM already configured")

# Fix observer.py to use Qwen3MoE hook config
print("\nConfiguring observer.py...")
with open('src/reap/observer.py', 'r') as f:
    content = f.read()

if 'Qwen3NextForCausalLM' not in content:
    # Add mapping to use Qwen3MoE observer (which should work for Qwen3Next)
    content = content.replace(
        'OBSERVER_CONFIG_REGISTRY = {',
        'OBSERVER_CONFIG_REGISTRY = {\n    "Qwen3NextForCausalLM": Qwen3MoEObserverHookConfig,'
    )
    with open('src/reap/observer.py', 'w') as f:
        f.write(content)
    print("✓ Added Qwen3NextForCausalLM observer mapping")
else:
    print("✓ Qwen3NextForCausalLM already in registry")

print("\n" + "="*60)
print("✓ Configuration complete!")
print("="*60)
EOF

echo ""
echo "Now test if it works:"
echo "  python src/reap/prune.py --model-name Qwen/Qwen3-Next-80B-A3B-Instruct --dataset-name theblackcat102/evol-codealpaca-v1 --seed 42 --samples_per_category 1 --run_observer_only true --record_pruning_metrics_only true --distance_measure cosine"

