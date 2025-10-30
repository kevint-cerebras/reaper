#!/bin/bash
# add_qwen3next_observer.sh - Add proper Qwen3Next observer configuration

set -e

cd /workspace/reaper
export PYTHONPATH="/workspace/reaper/src"

echo "Adding Qwen3Next observer configuration..."

python << 'EOF'
# Step 1: Add observer config class
print("Step 1: Adding Qwen3NextMoEObserverHookConfig class...")
with open('src/reap/observer.py', 'r') as f:
    content = f.read()

# Add the new observer config class after Qwen3MoEObserverHookConfig
if 'Qwen3NextMoEObserverHookConfig' not in content:
    new_config = '''

@dataclass
class Qwen3NextMoEObserverHookConfig(MoETransformerObserverConfig):
    module_class_name_to_hook_regex: Optional[str] = "Qwen3NextSparseMoeBlock"
'''
    
    # Insert after Qwen3MoEObserverHookConfig
    content = content.replace(
        'class Qwen3MoEObserverHookConfig(MoETransformerObserverConfig):\n    module_class_name_to_hook_regex: Optional[str] = "Qwen3MoeSparseMoeBlock"',
        'class Qwen3MoEObserverHookConfig(MoETransformerObserverConfig):\n    module_class_name_to_hook_regex: Optional[str] = "Qwen3MoeSparseMoeBlock"' + new_config
    )
    
    with open('src/reap/observer.py', 'w') as f:
        f.write(content)
    print("✓ Added Qwen3NextMoEObserverHookConfig class")
else:
    print("✓ Qwen3NextMoEObserverHookConfig already exists")

# Step 2: Register it in OBSERVER_CONFIG_REGISTRY
print("\nStep 2: Registering Qwen3NextForCausalLM in registry...")
with open('src/reap/observer.py', 'r') as f:
    content = f.read()

if '"Qwen3NextForCausalLM": Qwen3NextMoEObserverHookConfig' not in content:
    content = content.replace(
        'OBSERVER_CONFIG_REGISTRY = {',
        'OBSERVER_CONFIG_REGISTRY = {\n    "Qwen3NextForCausalLM": Qwen3NextMoEObserverHookConfig,'
    )
    with open('src/reap/observer.py', 'w') as f:
        f.write(content)
    print("✓ Registered Qwen3NextForCausalLM")
else:
    print("✓ Qwen3NextForCausalLM already registered")

# Step 3: Update model_util.py
print("\nStep 3: Configuring model_util.py...")
with open('src/reap/model_util.py', 'r') as f:
    content = f.read()

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
    print("✓ Added Qwen3NextForCausalLM to MODEL_ATTRS")
else:
    print("✓ Qwen3NextForCausalLM already in MODEL_ATTRS")

print("\n" + "="*60)
print("✓ Configuration complete!")
print("="*60)
print("\nNow you can run the full pipeline:")
print("  cd /workspace")
print("  ./run_qwen3next_pruning.sh")
EOF

