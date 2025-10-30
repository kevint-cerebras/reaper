# RunPod Pre-Flight Checklist

Complete this checklist BEFORE starting your 16-hour pruning run!

## ✅ 1. Account & Credits

- [ ] RunPod account created at https://runpod.io
- [ ] Payment method added
- [ ] **$200+ credits loaded** (recommended $250 for safety margin)
- [ ] Understand billing: charged per MINUTE of GPU time

**Cost Calculator:**
```
8x A100 40GB: ~$10-12/hour
16 hours runtime = $160-192
+ buffer for testing = $200-250 total
```

## ✅ 2. HuggingFace Account

- [ ] Account at https://huggingface.co
- [ ] **Token created** (Settings → Access Tokens → New Token → Read)
- [ ] Token saved securely: `hf_xxxxxxxxxxxxxxxxxxxxx`

**Why needed:**
- Faster downloads (no rate limits)
- Required to upload pruned models later
- Prevents download interruptions

## ✅ 3. GPU Selection on RunPod

**Recommended Configurations:**

| Option | GPUs | RAM | Disk | Cost/hr | Total Cost | Speed |
|--------|------|-----|------|---------|------------|-------|
| **Best** | 8x A100 40GB | 250GB+ | 500GB+ | $10-12 | $160-192 | Fast |
| Good | 4x A100 80GB | 300GB+ | 500GB+ | $12-15 | $192-240 | Medium |
| Overkill | 8x A100 80GB | 500GB+ | 1TB | $18-22 | $290-350 | Fastest |

**On RunPod:**
1. Go to "GPU Instances"
2. Filter: `A100` + `CUDA 12.1+`
3. Sort by: "Price (Low to High)"
4. Select pod with:
   - ✅ At least 4x A100 (prefer 8x)
   - ✅ 200GB+ RAM
   - ✅ 500GB+ disk
   - ✅ Good network (avoid "Community Cloud" if possible)

## ✅ 4. Template Selection

**Choose one:**

### Option A: PyTorch Template (Recommended)
```
runpod/pytorch:2.1.0-py3.10-cuda12.1.0-devel-ubuntu22.04
```
- Pre-installed: PyTorch, CUDA, Python
- Faster setup

### Option B: Base Template
```
runpod/base:0.4.0-cuda12.1.0
```
- Minimal bloat
- Slightly longer setup

## ✅ 5. Files Ready to Upload

Your local directory should have:

```
your-repo/
├── reap_prune.sh              ✓ Main script
├── test_reap.sh               ✓ Test script
├── validate_environment.sh    ✓ Validation
├── setup_runpod.sh            ✓ Initial setup
├── README.md                  ✓ Documentation
└── RUNPOD_CHECKLIST.md        ✓ This file
```

**All marked as executable:**
```bash
chmod +x *.sh
```

## ✅ 6. Connection Method

Choose your preferred method:

### Option A: SSH (Recommended for long runs)
- [ ] SSH keys generated locally
- [ ] SSH key added to RunPod (Account → Settings → Public Keys)
- [ ] Terminal tool ready (iTerm, Terminal, etc.)
- [ ] `tmux` or `screen` ready (keeps session alive)

### Option B: Web Terminal
- [ ] Understand browser must stay open
- [ ] Plan for 16+ hour browser session
- [ ] Consider using a dedicated browser profile

**Pro tip:** Use SSH + tmux for safety:
```bash
ssh root@your-pod-ip
tmux new -s reap
# Now if SSH disconnects, session continues!
# Reconnect: tmux attach -t reap
```

## ✅ 7. Monitoring Plan

- [ ] Monitoring dashboard open (RunPod web)
- [ ] Alerts configured (email/SMS for pod shutdown)
- [ ] Calendar blocked (don't schedule restarts during run)
- [ ] Phone/laptop charged for remote monitoring

**Monitoring commands ready:**
```bash
# GPU usage
watch -n 5 nvidia-smi

# Log monitoring
tail -f /workspace/reap_run.log

# Disk space
df -h /workspace
```

## ✅ 8. Backup Plan

- [ ] Know how to pause/resume pod (in emergency)
- [ ] Understand cost of paused pod (storage fees)
- [ ] Have backup credits if pod needs restart
- [ ] Screenshots of successful tests saved

## ✅ 9. Post-Processing Ready

- [ ] Plan for where to store 5 pruned models (~400GB total)
- [ ] HuggingFace repo created (if uploading)
- [ ] Storage space on local machine (if downloading)
- [ ] S3/GCS bucket ready (if cloud storage)

**Storage estimates:**
```
Each pruned model: 60-80GB
5 models total: ~400GB
Download time: 2-4 hours per model at 50MB/s
```

## ✅ 10. Time Planning

**Your 16-hour timeline:**

| Phase | Duration | Can Disconnect? | Notes |
|-------|----------|-----------------|-------|
| Upload files | 5 min | No | Initial setup |
| Setup & validation | 30 min | No | Run tests |
| Model download | 2-3 hrs | Yes* | Can monitor remotely |
| Observation | 8-12 hrs | Yes* | Longest phase |
| Pruning | 2 hrs | Yes* | Final stretch |
| Verification | 15 min | No | Check outputs |

*If using SSH + tmux

**Recommended schedule:**
```
Start: Evening (e.g., 8 PM)
Download: Overnight (8 PM - 11 PM)
Observation: Overnight + next day (11 PM - 11 AM)
Pruning: Midday (11 AM - 1 PM)
Done: Afternoon (1 PM next day)
```

## ✅ 11. Emergency Contacts

- [ ] RunPod support ticket system accessible
- [ ] REAP GitHub issues page bookmarked
- [ ] Discord/Slack channels for help ready
- [ ] This repo's URL saved for reference

## ✅ 12. Pre-Test Checklist

Before the actual run:

- [ ] `./validate_environment.sh` passes
- [ ] `./test_reap.sh` completes successfully (30 min)
- [ ] Test output verified: small model pruned correctly
- [ ] Logs look normal in `/workspace/reap_run.log`
- [ ] GPU utilization shows 80%+ during test
- [ ] Network speed adequate (download test model quickly)

## ✅ 13. Mental Preparation

- [ ] Understand this is a 16-hour commitment
- [ ] Okay with potentially losing $200 if something fails
- [ ] Plan B if pod crashes (restart with new pod)
- [ ] Excitement level: Maximum! 🚀

## 🚀 GO/NO-GO Decision

**Only start if ALL of these are true:**

- ✅ All checkboxes above are checked
- ✅ Test run completed successfully
- ✅ Have $200+ RunPod credits
- ✅ Next 16 hours are clear
- ✅ SSH + tmux configured (or willing to leave browser open)
- ✅ Disk space and RAM verified on pod
- ✅ HF token configured

## Launch Commands

Once all checks pass:

```bash
# On RunPod, in /workspace:

# 1. First time setup
./setup_runpod.sh
export HF_TOKEN='hf_your_token_here'

# 2. Validate
./validate_environment.sh

# 3. Test (~30 min)
./test_reap.sh

# 4. If test passes, GO!
tmux new -s reap
./reap_prune.sh

# 5. Detach from tmux: Ctrl+B then D
# 6. Monitor: tail -f /workspace/reap_run.log
# 7. Reconnect: tmux attach -t reap
```

## Timeline Tracker

Copy this to track progress:

```
[ ] Setup complete       - Expected: +30min
[ ] Download complete    - Expected: +3hrs  (Total: 3.5hrs)
[ ] Observation started  - Expected: +0min  (Total: 3.5hrs)
[ ] Observation 25%      - Expected: +3hrs  (Total: 6.5hrs)
[ ] Observation 50%      - Expected: +5hrs  (Total: 8.5hrs)
[ ] Observation 75%      - Expected: +8hrs  (Total: 11.5hrs)
[ ] Observation complete - Expected: +10hrs (Total: 13.5hrs)
[ ] 50% pruning done     - Expected: +30min (Total: 14hrs)
[ ] 30% pruning done     - Expected: +30min (Total: 14.5hrs)
[ ] 10% pruning done     - Expected: +30min (Total: 15hrs)
[ ] 5% pruning done      - Expected: +20min (Total: 15.3hrs)
[ ] 2% pruning done      - Expected: +20min (Total: 15.5hrs)
[ ] ALL COMPLETE! 🎉     - Expected: +10min (Total: ~16hrs)
```

---

**When ready, proceed to README.md for execution instructions!**

