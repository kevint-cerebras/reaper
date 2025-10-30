#!/bin/bash
# test_reap_v10.sh - Debug and fix Qwen2Moe observer config

set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "REAP Test - V10 (Debug)"
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

echo "→ Inspecting test model architecture..."
python << 'EOF'
from transformers import AutoModelForCausalLM
import torch

print("Loading model to inspect structure...")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen1.5-MoE-A2.7B",
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="cpu"
)

print(f"\nModel class: {model.__class__.__name__}")
print(f"\nModel structure (first layer):")
layer = model.model.layers[0]
print(f"Layer type: {type(layer)}")
print(f"Layer attributes: {dir(layer.mlp)}")

# Check if it has MoE
if hasattr(layer.mlp, 'experts'):
    print(f"\n✓ Has experts!")
    print(f"  Number of experts: {len(layer.mlp.experts)}")
    print(f"  Expert type: {type(layer.mlp.experts[0])}")
    print(f"  Expert attributes: {dir(layer.mlp.experts[0])}")
    
    # Check gate
    if hasattr(layer.mlp, 'gate'):
        print(f"  Gate type: {type(layer.mlp.gate)}")
    elif hasattr(layer.mlp, 'router'):
        print(f"  Router type: {type(layer.mlp.router)}")
else:
    print("\n✗ No experts found!")
    print(f"MLP attributes: {dir(layer.mlp)}")

# Check config
print(f"\nConfig:")
if hasattr(model.config, 'num_experts'):
    print(f"  num_experts: {model.config.num_experts}")
if hasattr(model.config, 'num_experts_per_tok'):
    print(f"  num_experts_per_tok: {model.config.num_experts_per_tok}")

del model
torch.cuda.empty_cache()
print("\n✓ Inspection complete")
EOF

echo ""
echo "Based on the architecture above, we'll configure the model..."
echo "Check the output and see if we need to adjust the config."

