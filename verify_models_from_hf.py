#!/usr/bin/env python3
"""
Verify REAP pruned models downloaded from HuggingFace
Tests inference and model integrity
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import sys

MODELS = {
    "2%": "kevint00/Qwen3-Next-80B-A3B-Instruct-REAP-2pct",
    "5%": "kevint00/Qwen3-Next-80B-A3B-Instruct-REAP-5pct",
    "10%": "kevint00/Qwen3-Next-80B-A3B-Instruct-REAP-10pct",
    "30%": "kevint00/Qwen3-Next-80B-A3B-Instruct-REAP-30pct",
    "50%": "kevint00/Qwen3-Next-80B-A3B-Instruct-REAP-50pct",
}

TEST_PROMPTS = [
    "Write a Python function to calculate fibonacci:\n\n```python\n",
    "Explain quantum computing:\n\n",
    "Complete this code:\ndef quicksort(arr):\n"
]

def test_model(name, repo_id):
    """Test a single model"""
    print(f"\n{'='*70}")
    print(f"Testing {name} - {repo_id}")
    print(f"{'='*70}")
    
    try:
        # Load model
        print("⏳ Loading model...")
        start = time.time()
        
        tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            repo_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        
        load_time = time.time() - start
        params = sum(p.numel() for p in model.parameters())
        
        print(f"✅ Loaded in {load_time:.1f}s")
        print(f"✅ Parameters: {params/1e9:.2f}B")
        
        # Count experts
        expert_count = 0
        for module in model.modules():
            if hasattr(module, 'experts') and hasattr(module.experts, '__len__'):
                expert_count += len(module.experts)
        
        if expert_count > 0:
            print(f"✅ Experts: {expert_count} total")
        
        # Test inference with GREEDY (no sampling - avoids router issues)
        print("\n🧪 Testing inference (greedy decoding)...")
        
        for i, prompt in enumerate(TEST_PROMPTS[:2], 1):  # Test 2 prompts
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            start = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,  # Greedy to avoid sampling issues
                    pad_token_id=tokenizer.eos_token_id
                )
            
            gen_time = time.time() - start
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated = response[len(prompt):]
            
            tokens = len(outputs[0]) - len(inputs.input_ids[0])
            tps = tokens / gen_time
            
            print(f"  ✅ Test {i}: {tps:.1f} tok/s ({gen_time:.2f}s)")
            print(f"     Output: {generated[:60]}...")
        
        # Cleanup
        del model, tokenizer
        torch.cuda.empty_cache()
        
        print(f"\n✅ {name} - PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ {name} - FAILED")
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*70)
    print("REAP PRUNED MODEL VERIFICATION FROM HUGGINGFACE")
    print("="*70)
    
    # Test each model (smallest to largest)
    results = {}
    for name in ["2%", "5%", "10%", "30%", "50%"]:
        repo_id = MODELS[name]
        results[name] = test_model(name, repo_id)
        
        # Small break between models
        time.sleep(2)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {name} ({MODELS[name]})")
    
    passed_count = sum(1 for v in results.values() if v)
    print(f"\n{passed_count}/{len(results)} models passed")
    
    if passed_count == len(results):
        print("\n🎉 All models working! Ready for benchmarking!")
        return 0
    else:
        print("\n⚠️  Some models failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

