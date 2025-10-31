#!/usr/bin/env python3
"""
Verify REAP pruned models: router structure and inference quality
Tests all 5 compression ratios (50%, 30%, 10%, 5%, 2%)
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import sys
from pathlib import Path
import time

# Configuration
BASE_PATH = "/workspace/reaper/artifacts/Qwen3-Next-80B-A3B-Instruct/evol-codealpaca-v1/all"
COMPRESSION_RATIOS = ["0.5", "0.3", "0.1", "0.05", "0.02"]
RATIO_NAMES = {
    "0.5": "50% (256 experts/layer)",
    "0.3": "30% (154 experts/layer)", 
    "0.1": "10% (51 experts/layer)",
    "0.05": "5% (26 experts/layer)",
    "0.02": "2% (10 experts/layer)"
}

# Test prompts
TEST_PROMPTS = [
    "Write a Python function to calculate fibonacci numbers:\n\n```python\n",
    "Explain quantum computing in simple terms:\n\n",
    "def bubble_sort(arr):\n    # Complete this sorting function\n"
]


def count_experts(model):
    """Count total and active experts in the model"""
    total_experts = 0
    layers_with_experts = 0
    
    for name, module in model.named_modules():
        # Check for Qwen3Next MoE blocks
        if hasattr(module, 'experts'):
            layers_with_experts += 1
            if hasattr(module.experts, '__len__'):
                num_experts = len(module.experts)
                total_experts += num_experts
                
    return total_experts, layers_with_experts


def test_router_structure(model, compression_ratio):
    """Verify router structure is intact"""
    print(f"\n{'='*60}")
    print(f"Testing {RATIO_NAMES[compression_ratio]}")
    print(f"{'='*60}")
    
    # Count experts
    total_experts, num_layers = count_experts(model)
    
    if total_experts == 0:
        print("❌ ERROR: No experts found in model!")
        return False
        
    avg_experts = total_experts / num_layers if num_layers > 0 else 0
    
    print(f"✅ Router structure intact:")
    print(f"   - Total layers with MoE: {num_layers}")
    print(f"   - Total experts: {total_experts}")
    print(f"   - Avg experts/layer: {avg_experts:.1f}")
    
    # Check for router gates
    router_found = False
    for name, module in model.named_modules():
        if hasattr(module, 'gate') or 'gate' in name.lower():
            router_found = True
            break
    
    if router_found:
        print(f"✅ Router gates found")
    else:
        print(f"⚠️  WARNING: No router gates detected")
    
    return True


def test_inference(model, tokenizer, compression_ratio):
    """Test inference quality with multiple prompts"""
    print(f"\n📝 Testing inference...")
    
    device = model.device
    all_passed = True
    
    for i, prompt in enumerate(TEST_PROMPTS, 1):
        try:
            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            # Generate
            start_time = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            gen_time = time.time() - start_time
            
            # Decode
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Basic quality checks
            is_valid = (
                len(response) > len(prompt) and  # Generated something
                not response.count(response[0]) > len(response) * 0.5  # Not just repeating
            )
            
            status = "✅" if is_valid else "❌"
            print(f"\n{status} Test {i}/3 ({gen_time:.2f}s):")
            print(f"   Prompt: {prompt[:50]}...")
            print(f"   Generated: {response[len(prompt):100]}...")
            
            if not is_valid:
                all_passed = False
                
        except Exception as e:
            print(f"❌ Test {i}/3 FAILED: {e}")
            all_passed = False
    
    return all_passed


def test_model(compression_ratio):
    """Test a single pruned model"""
    model_path = Path(BASE_PATH) / f"{compression_ratio}_reap_saliency"
    
    if not model_path.exists():
        print(f"\n❌ Model not found: {model_path}")
        return False
    
    print(f"\n{'='*70}")
    print(f"TESTING: {RATIO_NAMES[compression_ratio]}")
    print(f"Path: {model_path}")
    print(f"{'='*70}")
    
    try:
        # Load tokenizer
        print("\n⏳ Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        print("✅ Tokenizer loaded")
        
        # Load model
        print("\n⏳ Loading model (this may take a few minutes)...")
        start_time = time.time()
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        
        load_time = time.time() - start_time
        print(f"✅ Model loaded in {load_time:.1f}s")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✅ Total parameters: {total_params:,}")
        
        # Test router structure
        structure_ok = test_router_structure(model, compression_ratio)
        
        # Test inference
        inference_ok = test_inference(model, tokenizer, compression_ratio)
        
        # Clean up
        del model
        del tokenizer
        torch.cuda.empty_cache()
        
        # Overall result
        if structure_ok and inference_ok:
            print(f"\n✅ {RATIO_NAMES[compression_ratio]} - ALL TESTS PASSED")
            return True
        else:
            print(f"\n❌ {RATIO_NAMES[compression_ratio]} - SOME TESTS FAILED")
            return False
            
    except Exception as e:
        print(f"\n❌ ERROR testing {compression_ratio}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Test all pruned models"""
    print("\n" + "="*70)
    print("REAP PRUNED MODEL VERIFICATION")
    print("="*70)
    print(f"\nBase path: {BASE_PATH}")
    print(f"Testing {len(COMPRESSION_RATIOS)} models...")
    
    # Check if base path exists
    if not Path(BASE_PATH).exists():
        print(f"\n❌ ERROR: Base path not found: {BASE_PATH}")
        print("\nMake sure you're running this on the RunPod instance with the pruned models!")
        sys.exit(1)
    
    # Test each model
    results = {}
    for ratio in COMPRESSION_RATIOS:
        results[ratio] = test_model(ratio)
        print("\n" + "="*70 + "\n")
    
    # Summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for ratio, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {RATIO_NAMES[ratio]}")
    
    print(f"\n{passed}/{total} models passed all tests")
    
    if passed == total:
        print("\n🎉 All pruned models are working correctly!")
        return 0
    else:
        print("\n⚠️  Some models have issues - review output above")
        return 1


if __name__ == "__main__":
    sys.exit(main())

