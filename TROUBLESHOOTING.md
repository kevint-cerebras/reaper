# Troubleshooting Guide

Common issues and solutions for REAP pruning on RunPod.

## 🔥 Critical Issues

### Pod Crashes / Out of Memory

**Symptoms:**
- Pod suddenly stops responding
- `CUDA out of memory` errors
- Process killed by OOM

**Solutions:**
```bash
# 1. Reduce GPU count
# Edit reap_prune.sh, line 27:
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Use 4 GPUs instead of 8

# 2. Or get a pod with more RAM per GPU
# Need: 200GB+ total system RAM

# 3. Restart from checkpoint
./reap_prune.sh  # Resumes from last completed step
```

### Out of Disk Space

**Symptoms:**
```
No space left on device
OSError: [Errno 28]
```

**Solutions:**
```bash
# Check space
df -h /workspace

# Clean cache
rm -rf /workspace/.cache/huggingface/hub/*blobs*

# Or clean unused models
du -sh /workspace/.cache/huggingface/hub/* | sort -h
rm -rf /workspace/.cache/huggingface/hub/models--<old-model>

# Verify cleared
df -h /workspace
```

### Download Fails / Timeout

**Symptoms:**
```
Connection timeout
Failed to download
HTTPError: 503
```

**Solutions:**
```bash
# 1. Set HF token (increases rate limits)
export HF_TOKEN='hf_your_token_here'

# 2. Resume download (automatic retry)
./reap_prune.sh  # Script resumes downloads

# 3. Manual download via CLI
pip install -U huggingface_hub
huggingface-cli download Qwen/Qwen3-Next-80B-A3B-Instruct \
    --cache-dir /workspace/.cache/huggingface

# Then run script
./reap_prune.sh
```

## ⚠️ Common Issues

### Script Won't Execute

**Symptoms:**
```
bash: ./reap_prune.sh: Permission denied
```

**Solution:**
```bash
chmod +x *.sh
./reap_prune.sh
```

### Python Package Missing

**Symptoms:**
```
ModuleNotFoundError: No module named 'transformers'
ImportError: cannot import name 'XXX'
```

**Solution:**
```bash
cd /workspace/reaper/reap-source
source .venv/bin/activate

# Reinstall packages
pip install --upgrade pip
pip install transformers torch accelerate

# Or rebuild environment
rm -rf .venv
bash scripts/build.sh
```

### Git Clone Fails

**Symptoms:**
```
fatal: destination path 'reaper' already exists
```

**Solution:**
```bash
# Remove and reclone
rm -rf /workspace/reaper
cd /workspace
git clone https://github.com/cerebras/reaper.git
```

### Observation Phase Stuck

**Symptoms:**
- No log output for >30 minutes
- GPU utilization at 0%
- Process appears hung

**Diagnosis:**
```bash
# Check if process is running
ps aux | grep python

# Check GPU usage
nvidia-smi

# Check last log entry
tail -20 /workspace/reap_run.log
```

**Solutions:**
```bash
# 1. Give it more time (might be loading model)
# Wait 1 hour before assuming it's stuck

# 2. If truly stuck, kill and restart
pkill -9 python
./reap_prune.sh  # Resumes from last checkpoint
```

### CUDA Version Mismatch

**Symptoms:**
```
CUDA driver version is insufficient
RuntimeError: CUDA error
```

**Solution:**
```bash
# Check CUDA version
nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print(torch.version.cuda)"

# If mismatch, reinstall PyTorch
pip install torch --upgrade --index-url https://download.pytorch.org/whl/cu121
```

## 🐛 Debugging Issues

### Enable Verbose Logging

**Add to script before running:**
```bash
export PYTHONUNBUFFERED=1
export TORCH_DISTRIBUTED_DEBUG=INFO

# Run with verbose output
./reap_prune.sh 2>&1 | tee -a debug.log
```

### Check Memory Usage

```bash
# System memory
free -h

# GPU memory
nvidia-smi

# Process memory
ps aux --sort=-%mem | head -10

# Disk I/O
iostat -x 5
```

### Verify Model Downloaded

```bash
# Check model cache
ls -lh /workspace/.cache/huggingface/hub/models--Qwen--Qwen3-Next-80B-A3B-Instruct/

# Should see snapshots/ directory with files
# Total size should be ~160GB
```

### Test Individual Steps

```bash
cd /workspace/reaper/reap-source
source .venv/bin/activate

# Test observation only
python src/reap/prune.py \
    --model-name "Qwen/Qwen3-Next-80B-A3B-Instruct" \
    --dataset-name "theblackcat102/evol-codealpaca-v1" \
    --seed 42 \
    --samples_per_category 10 \
    --run_observer_only true \
    --record_pruning_metrics_only true

# Test pruning only (requires observation file)
python src/reap/prune.py \
    --model-name "Qwen/Qwen3-Next-80B-A3B-Instruct" \
    --dataset-name "theblackcat102/evol-codealpaca-v1" \
    --compression-ratio 0.5 \
    --prune-method reap \
    --seed 42 \
    --samples_per_category 10 \
    --do-eval false
```

## 📊 Performance Issues

### Observation Too Slow

**Expected:** ~8-12 hours for 1024 samples  
**If slower:** Check GPU utilization

```bash
# Should see 80-100% GPU utilization
nvidia-smi dmon -s u

# If low utilization, check:
# 1. Not enough GPUs used
# 2. Disk I/O bottleneck
# 3. Network bottleneck (if loading from cache)
```

**Solutions:**
```bash
# 1. Use more GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 2. Reduce samples (faster but less accurate)
# Edit SAMPLES=512 in reap_prune.sh

# 3. Use faster disk
# Move cache to NVMe if available
```

### Download Too Slow

**Expected:** 2-3 hours for 160GB  
**If slower:** Check network speed

```bash
# Test download speed
wget -O /dev/null http://speedtest.tele2.net/100MB.zip

# Should see >50MB/s
# If <20MB/s, pod has slow network
```

**Solution:**
```bash
# Request different pod with better network
# Or wait it out (will complete eventually)
```

## 🚨 Error Messages Decoded

### `RuntimeError: CUDA out of memory`

**Meaning:** Not enough GPU memory  
**Fix:** Use fewer GPUs or get GPUs with more memory

### `OSError: [Errno 28] No space left on device`

**Meaning:** Disk full  
**Fix:** Clean cache (see "Out of Disk Space" above)

### `FileNotFoundError: observations_*.pt`

**Meaning:** Observation file missing/corrupted  
**Fix:** Delete artifacts/ and run observation again

### `ValueError: Unsupported architecture`

**Meaning:** Qwen3-Next not configured  
**Fix:** Check that `setup_runpod.sh` ran successfully

### `ConnectionError: Connection timed out`

**Meaning:** Network issue downloading model  
**Fix:** Retry script (automatic resume)

### `ImportError: cannot import name 'XXX' from 'transformers'`

**Meaning:** Wrong transformers version  
**Fix:** `pip install transformers==4.55.0`

## 🔧 Recovery Procedures

### Full Reset (Nuclear Option)

```bash
# Stop everything
pkill -9 python

# Clean workspace
cd /workspace
rm -rf reaper/
rm -rf .cache/

# Start fresh
./reap_prune.sh
```

### Partial Reset (Keep Downloads)

```bash
# Stop processes
pkill -9 python

# Keep cache, remove artifacts
cd /workspace/reaper/reap-source
rm -rf artifacts/

# Restart (uses cached model)
cd /workspace
./reap_prune.sh
```

### Resume After Interruption

```bash
# Script automatically detects completed steps
# Just run again:
./reap_prune.sh

# It will skip:
# - Downloaded model
# - Completed observations
# - Already pruned variants
```

## 📞 Getting Help

### Check Logs First

```bash
# Main log
tail -100 /workspace/reap_run.log

# Python errors
grep -i "error\|exception\|traceback" /workspace/reap_run.log

# Last successful step
grep "✓" /workspace/reap_run.log | tail -5
```

### Gather Debug Info

```bash
# Create debug report
cat > /workspace/debug_report.txt << EOF
GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
CUDA: $(python -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "N/A")
Disk: $(df -h /workspace | tail -1)
RAM: $(free -h | grep Mem | awk '{print $3 "/" $2}')
Last Error: $(grep -i error /workspace/reap_run.log | tail -1)
EOF

cat /workspace/debug_report.txt
```

### Where to Ask

1. **REAP GitHub Issues:** https://github.com/cerebras/reaper/issues
2. **RunPod Discord:** Community support channel
3. **HuggingFace Forums:** For model-specific issues

**Include in your request:**
- Debug report (from above)
- Last 50 lines of log: `tail -50 /workspace/reap_run.log`
- GPU configuration
- What step it failed on

## ✅ Prevention Checklist

**Before starting:**
- [ ] Run `./validate_environment.sh` 
- [ ] Run `./test_reap.sh` successfully
- [ ] Have 500GB+ free disk space
- [ ] Have 200GB+ RAM
- [ ] Have stable network connection
- [ ] Using SSH + tmux (not just web terminal)
- [ ] $200+ RunPod credits available

**This prevents 90% of issues!**

---

**Still stuck? Create an issue in this repo with your debug report!**

