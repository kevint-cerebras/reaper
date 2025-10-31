# Local Inference Testing Guide

Test all 5 REAP pruned Qwen3-Next models on your local machine.

## Requirements

```bash
pip install torch transformers accelerate
```

**GPU Requirements:**
- Minimum: 40GB VRAM (A100, H100)
- Recommended: 80GB VRAM for comfortable testing
- Models are loaded in `bfloat16` format

## Quick Start

### Test all models from HuggingFace:
```bash
python local_inference_test.py
```

The script will automatically download models from HuggingFace on first run.

### Test specific models:
```bash
# Test only the smallest models
python local_inference_test.py --models 2pct 5pct

# Test medium compression
python local_inference_test.py --models 50pct 30pct
```

### Use locally downloaded models:
```bash
# If you've already downloaded models to a local directory
python local_inference_test.py --local-path /path/to/models/
```

### Generate more tokens:
```bash
# Generate 200 tokens per prompt instead of 100
python local_inference_test.py --max-tokens 200
```

## Available Models

| Model | Compression | Experts/Layer | Approx Size |
|-------|-------------|---------------|-------------|
| `50pct` | 50% | 256 | ~80-90 GB |
| `30pct` | 30% | 154 | ~50-60 GB |
| `10pct` | 10% | 51 | ~20-25 GB |
| `5pct` | 5% | 26 | ~10-15 GB |
| `2pct` | 2% | 10 | ~5-8 GB |

## What the script tests

For each model:
1. ✅ **Load time** - How fast the model loads
2. ✅ **Parameter count** - Verify compression worked
3. ✅ **Code generation** - Generate a Python function
4. ✅ **Explanation** - Explain a technical concept
5. ✅ **Code completion** - Complete a code snippet
6. ✅ **Generation speed** - Tokens per second

## Example Output

```
======================================================================
TESTING: 50pct
======================================================================

⏳ Loading model from: kevint00/Qwen3-Next-80B-A3B-Instruct-REAP-50pct
✅ Loaded in 45.2s | 42.5B params

📝 Code Generation:
   Prompt: Write a Python function to calculate fibonacci...
   ⚡ 3.2s | 31.2 tok/s
   📄 Output: def fibonacci(n):
       if n <= 1:
           return n
       return fibonacci(n-1) + fibonacci(n-2)
   ...

✅ 50pct - All tests passed
```

## Tips

- **First run will be slow** - models need to download from HuggingFace
- **Use `--models 2pct` first** - test with the smallest model to verify setup
- **Monitor GPU memory** - use `nvidia-smi` to check VRAM usage
- **Local paths save time** - download once, test multiple times

## Troubleshooting

### Out of Memory
```bash
# Test only the smallest model
python local_inference_test.py --models 2pct

# Or generate fewer tokens
python local_inference_test.py --models 5pct --max-tokens 50
```

### Slow downloads
Models are large (5-90GB each). Consider:
- Download overnight
- Use fast internet connection
- Or test on RunPod/cloud instance first

### Model not found
Make sure models are uploaded to HuggingFace:
- https://huggingface.co/kevint00/Qwen3-Next-80B-A3B-Instruct-REAP-50pct
- https://huggingface.co/kevint00/Qwen3-Next-80B-A3B-Instruct-REAP-30pct
- etc.

## Advanced Usage

### Python API

```python
from local_inference_test import load_model, run_inference

# Load model
model, tokenizer = load_model("kevint00/Qwen3-Next-80B-A3B-Instruct-REAP-2pct")

# Run inference
prompt = "Write a hello world program in Python:\n"
response, gen_time, tokens_per_sec = run_inference(model, tokenizer, prompt)

print(f"Generated in {gen_time:.2f}s at {tokens_per_sec:.1f} tok/s")
print(response)
```

## Questions?

- Check the main README.md
- See TROUBLESHOOTING.md for common issues

