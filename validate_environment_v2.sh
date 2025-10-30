#!/bin/bash
# validate_environment.sh - Pre-flight checks for RunPod environment

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "RunPod Environment Validation"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

ERRORS=0

# Check GPUs
echo "Checking GPUs..."
if nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    echo "✓ Found $GPU_COUNT GPUs"
    echo "  GPU 0 Memory: ${GPU_MEMORY}MB"
    
    if [ $GPU_COUNT -lt 4 ]; then
        echo "⚠  Warning: Less than 4 GPUs (have $GPU_COUNT)"
        echo "   Recommended: 8x A100 40GB or 4x A100 80GB"
    fi
else
    echo "✗ No NVIDIA GPUs found"
    ERRORS=$((ERRORS + 1))
fi

echo ""

# Check disk space
echo "Checking disk space..."
AVAILABLE_GB=$(df -BG /workspace | tail -1 | awk '{print $4}' | sed 's/G//')
echo "  Available: ${AVAILABLE_GB}GB"

if [ $AVAILABLE_GB -lt 400 ]; then
    echo "✗ Insufficient disk space (need 400GB+)"
    ERRORS=$((ERRORS + 1))
else
    echo "✓ Sufficient disk space"
fi

echo ""

# Check RAM
echo "Checking RAM..."
RAM_GB=$(free -g | awk '/^Mem:/{print $2}')
echo "  Total RAM: ${RAM_GB}GB"

if [ $RAM_GB -lt 150 ]; then
    echo "⚠  Low RAM: ${RAM_GB}GB (recommended 200GB+)"
else
    echo "✓ Sufficient RAM"
fi

echo ""

# Check internet
echo "Checking internet connection..."
if curl -s --connect-timeout 5 https://huggingface.co > /dev/null; then
    echo "✓ Internet connection active"
else
    echo "✗ No internet connection"
    ERRORS=$((ERRORS + 1))
fi

echo ""

# Check Python
echo "Checking Python..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | awk '{print $2}')
    echo "✓ Python $PYTHON_VERSION found"
else
    echo "✗ Python not found"
    ERRORS=$((ERRORS + 1))
fi

echo ""

# Check CUDA
echo "Checking CUDA..."
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep release | awk '{print $5}' | sed 's/,//')
    echo "✓ CUDA $CUDA_VERSION found"
else
    echo "⚠  nvcc not found (might be ok if CUDA is installed differently)"
fi

echo ""

# Summary
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ $ERRORS -eq 0 ]; then
    echo "✓ Environment validation passed"
    echo ""
    echo "Ready to run REAP pruning!"
    echo ""
    echo "Next steps:"
    echo "  1. Test with small model: ./test_reap.sh"
    echo "  2. Run full pipeline: ./reap_prune.sh"
    exit 0
else
    echo "✗ Environment validation failed ($ERRORS errors)"
    echo ""
    echo "Fix the errors above before proceeding"
    exit 1
fi



