#!/usr/bin/env python3
"""
Test REAP pruned models locally on RunPod before uploading
Uses models from /workspace/reaper/artifacts/
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
from pathlib import Path
import sys

# Local paths on RunPod
BASE_PATH = "/workspace/reaper/artifacts/Qwen3-Next-80B-A3B-Instruct/evol-codealpaca-v1/all"

MODELS = {
    "50%": "0.5_reap_saliency",
    "30%": "0.3_reap_saliency",
    "10%": "0.1_reap_saliency",
    "5%": "0.05_reap_saliency",
    "2%": "0.02_reap_saliency",
}

# Quick test prompts
PROMPTS = [
    "Write a Python function to calculate fibonacci numbers:\n\n```python\n",
    "Explain quantum computing in simple terms:\n\n",
    "def quicksort(arr):\n    # Complete this function\n"
]


def test_model(compression_name, model_dir, quick=True):
    """Test a single pruned model"""
    model_path = Path(BASE_PATH) / model_dir
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return False
    
    print(f"\n{'='*70}")
    print(f"TESTING: {compression_name} compression")
    print(f"Path: {model_path}")
    print(f"{'='*70}")
    
    try:
        # Load model
        print("\n⏳ Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            trust_remote_code=True
        )
        
        print("⏳ Loading model (may take 1-2 minutes)...")
        start = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        load_time = time.time() - start
        
        # Model info
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✅ Loaded in {load_time:.1f}s")
        print(f"✅ Parameters: {total_params/1e9:.2f}B")
        
        # Check MoE structure
        expert_count = 0
        for name, module in model.named_modules():
            if hasattr(module, 'experts'):
                if hasattr(module.experts, '__len__'):
                    expert_count += len(module.experts)
        
        if expert_count > 0:
            print(f"✅ Total experts: {expert_count}")
        
        # Run inference tests
        print(f"\n📝 Testing inference ({len(PROMPTS)} prompts)...")
        
        max_tokens = 50 if quick else 100
        
        for i, prompt in enumerate(PROMPTS, 1):
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            start = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            gen_time = time.time() - start
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated = response[len(prompt):]
            tokens_generated = len(outputs[0]) - len(inputs.input_ids[0])
            tps = tokens_generated / gen_time
            
            print(f"\n✅ Test {i}/{len(PROMPTS)}")
            print(f"   Speed: {tps:.1f} tok/s ({gen_time:.2f}s)")
            print(f"   Output: {generated[:80]}...")
        
        # Cleanup
        del model, tokenizer
        torch.cuda.empty_cache()
        
        print(f"\n✅ {compression_name} - ALL TESTS PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*70)
    print("LOCAL REAP PRUNED MODEL TESTING")
    print("="*70)
    print(f"\nBase path: {BASE_PATH}\n")
    
    # Check if base path exists
    if not Path(BASE_PATH).exists():
        print(f"❌ ERROR: Artifacts directory not found!")
        print(f"Expected: {BASE_PATH}")
        print("\nMake sure you're on the RunPod instance with pruned models.")
        sys.exit(1)
    
    # Check which models exist
    print("Checking for pruned models...")
    available_models = []
    for name, dir_name in MODELS.items():
        model_path = Path(BASE_PATH) / dir_name
        if model_path.exists():
            print(f"  ✅ {name} - {dir_name}")
            available_models.append((name, dir_name))
        else:
            print(f"  ❌ {name} - NOT FOUND")
    
    if not available_models:
        print("\n❌ No pruned models found!")
        sys.exit(1)
    
    print(f"\nFound {len(available_models)} models to test\n")
    
    # Test each model
    results = {}
    for name, dir_name in available_models:
        success = test_model(name, dir_name, quick=True)
        results[name] = success
        
        # Give GPU a break between models
        time.sleep(2)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    for name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {name}")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"\n{passed}/{total} models passed")
    
    if passed == total:
        print("\n🎉 All models working! Ready to upload to HuggingFace!")
        return 0
    else:
        print("\n⚠️  Some models failed - DO NOT upload yet")
        return 1


if __name__ == "__main__":
    sys.exit(main())

