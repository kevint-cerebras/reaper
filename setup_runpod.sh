#!/bin/bash
# setup_runpod.sh - Initial RunPod configuration

set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "RunPod Initial Setup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Set HuggingFace token (if provided)
if [ -z "$HF_TOKEN" ]; then
    echo "⚠  HF_TOKEN not set"
    echo "   To set it: export HF_TOKEN='hf_your_token_here'"
    echo "   (Optional but recommended for faster downloads)"
else
    echo "✓ HF_TOKEN is set"
    
    # Configure git credentials for HF
    git config --global credential.helper store
    echo "https://user:${HF_TOKEN}@huggingface.co" > ~/.git-credentials
    
    # Login to HF CLI
    pip install -q huggingface_hub
    huggingface-cli login --token $HF_TOKEN
    
    echo "✓ HuggingFace configured"
fi

echo ""

# Install system dependencies
echo "Installing system dependencies..."
apt-get update -qq
apt-get install -y -qq wget curl git build-essential bc > /dev/null 2>&1
echo "✓ System dependencies installed"

echo ""

# Setup workspace structure
echo "Setting up workspace..."
mkdir -p /workspace/.cache/huggingface
mkdir -p /workspace/logs
mkdir -p /workspace/scripts

echo "✓ Workspace structure created"

echo ""

# Set up monitoring
echo "Creating monitoring helpers..."

# GPU monitor script
cat > /workspace/scripts/monitor_gpu.sh << 'EOF'
#!/bin/bash
watch -n 5 'nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv'
EOF

# Log monitor script
cat > /workspace/scripts/monitor_log.sh << 'EOF'
#!/bin/bash
tail -f /workspace/reap_run.log
EOF

# Disk space monitor
cat > /workspace/scripts/check_space.sh << 'EOF'
#!/bin/bash
echo "Disk Usage:"
df -h /workspace
echo ""
echo "Largest directories:"
du -h -d 1 /workspace 2>/dev/null | sort -hr | head -10
EOF

chmod +x /workspace/scripts/*.sh

echo "✓ Monitoring scripts created"

echo ""

# Create quick reference card
cat > /workspace/QUICK_REFERENCE.txt << 'EOF'
═══════════════════════════════════════════════════════════════
                    QUICK REFERENCE
═══════════════════════════════════════════════════════════════

MAIN COMMANDS:
  ./validate_environment.sh    - Check system requirements
  ./test_reap.sh              - Test with small model (~30 min)
  ./reap_prune.sh             - Run full pipeline (~16 hours)

MONITORING:
  tail -f /workspace/reap_run.log              - Watch main log
  /workspace/scripts/monitor_gpu.sh            - GPU usage
  /workspace/scripts/monitor_log.sh            - Formatted log
  /workspace/scripts/check_space.sh            - Disk usage

DURING RUN:
  • DON'T close terminal/browser (keeps SSH alive)
  • DON'T stop the pod
  • DON'T run multiple instances
  • DO monitor disk space regularly

IF INTERRUPTED:
  • Just restart: ./reap_prune.sh
  • Script resumes from last checkpoint
  • Already completed steps are skipped

TIMELINE:
  Setup:        30 min
  Download:     2-3 hours
  Observation:  8-12 hours (happens ONCE)
  Pruning:      2 hours (all 5 variants)
  Total:        ~14-16 hours

OUTPUT:
  /workspace/reaper/artifacts/
    └── Qwen3-Next-80B-A3B-Instruct/
        └── evol-codealpaca-v1/
            └── pruned_models/
                ├── reap-seed_42-0.5/   (50%)
                ├── reap-seed_42-0.7/   (30%)
                ├── reap-seed_42-0.9/   (10%)
                ├── reap-seed_42-0.95/  (5%)
                └── reap-seed_42-0.98/  (2%)

COST ESTIMATE:
  8x A100 40GB: $10-12/hr × 16hrs = $160-190
  
EMERGENCY:
  • Out of space? rm -rf /workspace/.cache/huggingface/hub/*blobs*
  • Out of memory? Restart pod with more RAM
  • Script stuck? Check: ps aux | grep python

═══════════════════════════════════════════════════════════════
EOF

echo "✓ Quick reference created: /workspace/QUICK_REFERENCE.txt"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✓ RunPod setup complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Next steps:"
echo "  1. cat /workspace/QUICK_REFERENCE.txt"
echo "  2. ./validate_environment.sh"
echo "  3. ./test_reap.sh"
echo "  4. ./reap_prune.sh"
echo ""

