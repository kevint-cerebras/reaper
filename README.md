# REAP Pruning for Qwen3-Next-80B-A3B-Instruct

Prune Qwen3-Next-80B-A3B from 24,576 experts to 5 different compression levels using REAP.

## 📋 Before You Start

**REQUIRED READING:**
1. **[RUNPOD_CHECKLIST.md](RUNPOD_CHECKLIST.md)** - Complete this first! ⚠️
2. **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - Bookmark for issues

## Quick Start

```bash
# 0. FIRST TIME ONLY: Initial setup
./setup_runpod.sh
export HF_TOKEN='hf_your_token_here'  # Optional but recommended

# 1. Validate environment
./validate_environment.sh

# 2. Test with small model (~30 min)
./test_reap.sh

# 3. If test passes, run full pipeline (~14-16 hours)
tmux new -s reap
./reap_prune.sh

# 4. Detach from tmux (keeps running): Ctrl+B then D
# 5. Monitor: tail -f /workspace/reap_run.log
# 6. Reattach: tmux attach -t reap
```

## What You Get

5 pruned models at different compression levels:

| Variant | Experts/Layer | Total Experts | Router Ratio | Description |
|---------|--------------|---------------|--------------|-------------|
| 50% | 256 | 12,288 | 25.6:1 | Production ready |
| 30% | 154 | 7,392 | 15.4:1 | Resource constrained |
| 10% | 51 | 2,448 | 5.1:1 | Research/extreme edge |
| 5% | 26 | 1,248 | 2.6:1 | Nuclear territory |
| 2% | 10 | 480 | 1.0:1 | Comedy science 🎪 |

## Timeline

- **Setup**: 30 min (environment, dependencies, model download prep)
- **Model Download**: 2-3 hours (~160GB)
- **Observation**: 8-12 hours (recording 24,576 expert activations - happens ONCE)
- **Pruning**: 2 hours total (all 5 variants)
- **Total**: ~14-16 hours

## Output Location

```
/workspace/reaper/reap-source/artifacts/
└── Qwen3-Next-80B-A3B-Instruct/
    └── evol-codealpaca-v1/
        └── pruned_models/
            ├── reap-seed_42-0.5/    (50% - 256 experts/layer)
            ├── reap-seed_42-0.7/    (30% - 154 experts/layer)
            ├── reap-seed_42-0.9/    (10% - 51 experts/layer)
            ├── reap-seed_42-0.95/   (5% - 26 experts/layer)
            └── reap-seed_42-0.98/   (2% - 10 experts/layer)
```

## Monitoring

```bash
# Watch the log in real-time
tail -f /workspace/reap_run.log

# Check GPU usage
nvidia-smi

# Check what's running
ps aux | grep python

# Check disk usage
df -h /workspace
```

## Requirements

- **GPUs**: 8x A100 40GB or 4x A100 80GB (minimum 4x A100)
- **RAM**: 200GB+ recommended (150GB minimum)
- **Disk**: 500GB+ (400GB minimum)
- **Time**: ~16 hours for complete pipeline

## Resume Capability

The script automatically skips completed steps:
- ✓ If model is cached, skips download
- ✓ If observations exist, skips observation phase
- ✓ If variant is pruned, skips that compression

You can safely restart after interruption!

## Troubleshooting

**Out of disk space:**
```bash
# Clean up cache if needed
rm -rf /workspace/.cache/huggingface/hub/*blobs*
```

**Out of memory:**
```bash
# Reduce GPU count in reap_prune.sh
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Use only 4 GPUs
```

**Script fails:**
```bash
# Check the log
tail -100 /workspace/reap_run.log

# The script shows exactly which line failed
```

## 📁 Files in This Package

**Scripts:**
- `setup_runpod.sh` - Initial RunPod configuration (run once)
- `reap_prune.sh` - Main pruning pipeline (production-ready)
- `test_reap.sh` - Test with small model (~30 min)
- `validate_environment.sh` - Environment checks (instant)

**Documentation:**
- `README.md` - This file (usage guide)
- `RUNPOD_CHECKLIST.md` - **Pre-flight checklist** (complete before starting!)
- `TROUBLESHOOTING.md` - Solutions for common issues

## Next Steps After Pruning

1. **Test the models:**
```bash
cd /workspace/reaper/reap-source
source .venv/bin/activate

# Test generation
python << EOF
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = "artifacts/Qwen3-Next-80B-A3B-Instruct/evol-codealpaca-v1/pruned_models/reap-seed_42-0.5"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

prompt = "Write a Python function to calculate fibonacci:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
EOF
```

2. **Upload to HuggingFace:**
```bash
# Install huggingface_hub
pip install huggingface_hub

# Login
huggingface-cli login

# Upload
huggingface-cli upload your-username/Qwen3-Next-50pct-REAP \
    artifacts/Qwen3-Next-80B-A3B-Instruct/evol-codealpaca-v1/pruned_models/reap-seed_42-0.5
```

3. **Deploy with vLLM:**
```bash
vllm serve artifacts/Qwen3-Next-80B-A3B-Instruct/evol-codealpaca-v1/pruned_models/reap-seed_42-0.5 \
    --trust-remote-code
```

## Citation

```bibtex
@misc{lasby-reap,
    title={REAP the Experts: Why Pruning Prevails for One-Shot MoE compression},
    author={Lasby, Mike and Lazarevich, Ivan and Sinnadurai, Nish and Lie, Sean and Ioannou, Yani and Thangarasa, Vithursan},
    year={2025},
    url={https://arxiv.org/abs/2510.13999}
}

@misc{qwen3technicalreport,
    title={Qwen3 Technical Report},
    author={Qwen Team},
    year={2025},
    url={https://arxiv.org/abs/2505.09388}
}
```



