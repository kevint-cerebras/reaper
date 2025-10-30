#!/bin/bash
# inspect_qwen3next.sh - Inspect Qwen3-Next-80B architecture to fix observer config

set -e

cd /workspace/reaper
export PYTHONPATH="/workspace/reaper/src"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Inspecting Qwen3-Next-80B Architecture"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python << 'EOF'
from transformers import AutoModelForCausalLM, AutoConfig
import torch

print("Loading model config...")
config = AutoConfig.from_pretrained("Qwen/Qwen3-Next-80B-A3B-Instruct", trust_remote_code=True)
print(f"✓ Model class: {config.model_type}")
print(f"  Num layers: {config.num_hidden_layers}")
print(f"  Num experts: {getattr(config, 'num_experts', 'N/A')}")
print(f"  Experts per tok: {getattr(config, 'num_experts_per_tok', 'N/A')}")

print("\nLoading first layer only to inspect structure...")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-Next-80B-A3B-Instruct",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    low_cpu_mem_usage=True
)

print(f"\n✓ Model class: {model.__class__.__name__}")

layer = model.model.layers[0]
print(f"\n=== Layer 0 Structure ===")
print(f"Layer type: {type(layer).__name__}")
print(f"Layer attributes: {[a for a in dir(layer) if not a.startswith('_')][:20]}")

if hasattr(layer, 'mlp'):
    mlp = layer.mlp
    print(f"\n=== MLP Structure ===")
    print(f"MLP type: {type(mlp).__name__}")
    mlp_attrs = [a for a in dir(mlp) if not a.startswith('_')]
    print(f"MLP attributes ({len(mlp_attrs)}): {mlp_attrs}")
    
    # Check for MoE components
    has_experts = hasattr(mlp, 'experts')
    has_gate = hasattr(mlp, 'gate')
    has_router = hasattr(mlp, 'router')
    has_shared = hasattr(mlp, 'shared_expert')
    
    print(f"\n=== MoE Components ===")
    print(f"Has experts: {has_experts}")
    print(f"Has gate: {has_gate}")
    print(f"Has router: {has_router}")
    print(f"Has shared_expert: {has_shared}")
    
    if has_experts:
        experts = mlp.experts
        print(f"\nExperts type: {type(experts).__name__}")
        if hasattr(experts, '__len__'):
            print(f"Number of experts: {len(experts)}")
        if hasattr(experts, '__getitem__'):
            try:
                expert0 = experts[0]
                print(f"Expert[0] type: {type(expert0).__name__}")
                expert_attrs = [a for a in dir(expert0) if not a.startswith('_')]
                print(f"Expert attributes: {expert_attrs}")
            except:
                print("Could not access expert[0]")
    
    if has_gate:
        print(f"Gate type: {type(mlp.gate).__name__}")
    
    if has_router:
        print(f"Router type: {type(mlp.router).__name__}")
    
    if has_shared:
        print(f"Shared expert type: {type(mlp.shared_expert).__name__}")

print("\n=== Summary ===")
print(f"Model: {model.__class__.__name__}")
print(f"MLP: {type(layer.mlp).__name__}")
if has_experts:
    print(f"Experts found: YES")
    print(f"Expert container: {type(mlp.experts).__name__}")
else:
    print(f"Experts found: NO - Not a MoE layer")

print("\n✓ Inspection complete")
print("\nCopy this output and share it so we can create the correct observer config!")

del model
torch.cuda.empty_cache()
EOF

