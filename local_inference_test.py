#!/usr/bin/env python3
"""
Local inference testing for REAP pruned Qwen3-Next models
Tests all 5 compression ratios: 50%, 30%, 10%, 5%, 2%
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
from pathlib import Path
import argparse

# Model configurations
MODELS = {
    "50pct": "kevint00/Qwen3-Next-80B-A3B-Instruct-REAP-50pct",
    "30pct": "kevint00/Qwen3-Next-80B-A3B-Instruct-REAP-30pct",
    "10pct": "kevint00/Qwen3-Next-80B-A3B-Instruct-REAP-10pct",
    "5pct": "kevint00/Qwen3-Next-80B-A3B-Instruct-REAP-5pct",
    "2pct": "kevint00/Qwen3-Next-80B-A3B-Instruct-REAP-2pct",
}

# Test prompts
PROMPTS = [
    {
        "name": "Code Generation",
        "text": "Write a Python function to calculate fibonacci numbers:\n\n```python\n"
    },
    {
        "name": "Explanation",
        "text": "Explain how transformers work in machine learning:\n\n"
    },
    {
        "name": "Code Completion",
        "text": "def merge_sort(arr):\n    # Complete this sorting algorithm\n"
    }
]


def load_model(model_id, use_local_path=None):
    """Load model and tokenizer"""
    path = use_local_path if use_local_path else model_id
    
    print(f"\n⏳ Loading model from: {path}")
    start = time.time()
    
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    load_time = time.time() - start
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"✅ Loaded in {load_time:.1f}s | {total_params/1e9:.2f}B params")
    
    return model, tokenizer


def run_inference(model, tokenizer, prompt_text, max_tokens=100):
    """Run single inference"""
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    
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
    tokens_generated = len(outputs[0]) - len(inputs.input_ids[0])
    tokens_per_sec = tokens_generated / gen_time
    
    return response, gen_time, tokens_per_sec


def test_model(model_name, model_id, use_local_path=None, max_tokens=100):
    """Test a single model"""
    print("\n" + "="*70)
    print(f"TESTING: {model_name}")
    print("="*70)
    
    try:
        # Load model
        model, tokenizer = load_model(model_id, use_local_path)
        
        # Run inference on all prompts
        results = []
        for prompt in PROMPTS:
            print(f"\n📝 {prompt['name']}:")
            print(f"   Prompt: {prompt['text'][:50]}...")
            
            response, gen_time, tps = run_inference(
                model, tokenizer, prompt['text'], max_tokens
            )
            
            # Extract generated part
            generated = response[len(prompt['text']):]
            
            print(f"   ⚡ {gen_time:.2f}s | {tps:.1f} tok/s")
            print(f"   📄 Output: {generated[:100]}...")
            
            results.append({
                "prompt": prompt['name'],
                "time": gen_time,
                "tokens_per_sec": tps,
                "output": generated[:200]
            })
        
        # Cleanup
        del model, tokenizer
        torch.cuda.empty_cache()
        
        print(f"\n✅ {model_name} - All tests passed")
        return True, results
        
    except Exception as e:
        print(f"\n❌ {model_name} - Failed: {e}")
        import traceback
        traceback.print_exc()
        return False, []


def main():
    parser = argparse.ArgumentParser(description="Test REAP pruned models")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODELS.keys()) + ["all"],
        default=["all"],
        help="Which models to test (default: all)"
    )
    parser.add_argument(
        "--local-path",
        type=str,
        help="Local directory containing downloaded models"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Max tokens to generate per prompt (default: 100)"
    )
    
    args = parser.parse_args()
    
    # Determine which models to test
    if "all" in args.models:
        models_to_test = MODELS.items()
    else:
        models_to_test = [(k, MODELS[k]) for k in args.models]
    
    print("\n" + "="*70)
    print("REAP PRUNED MODEL INFERENCE TEST")
    print("="*70)
    print(f"\nTesting {len(models_to_test)} models")
    print(f"Max tokens per prompt: {args.max_tokens}")
    
    if args.local_path:
        print(f"Using local models from: {args.local_path}")
    else:
        print("Downloading models from HuggingFace (may take a while first time)")
    
    # Test each model
    all_results = {}
    for model_name, model_id in models_to_test:
        # Handle local path
        local_path = None
        if args.local_path:
            local_path = Path(args.local_path) / model_id.split("/")[-1]
            if not local_path.exists():
                print(f"⚠️  Local path not found: {local_path}, using HuggingFace")
                local_path = None
        
        success, results = test_model(
            model_name,
            model_id,
            use_local_path=str(local_path) if local_path else None,
            max_tokens=args.max_tokens
        )
        
        all_results[model_name] = {
            "success": success,
            "results": results
        }
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    for model_name, data in all_results.items():
        status = "✅ PASS" if data["success"] else "❌ FAIL"
        print(f"\n{status} - {model_name}")
        
        if data["success"] and data["results"]:
            avg_tps = sum(r["tokens_per_sec"] for r in data["results"]) / len(data["results"])
            print(f"   Average speed: {avg_tps:.1f} tokens/sec")
    
    # Overall stats
    passed = sum(1 for d in all_results.values() if d["success"])
    total = len(all_results)
    print(f"\n{passed}/{total} models passed")
    
    if passed == total:
        print("\n🎉 All models working correctly!")
    else:
        print("\n⚠️  Some models failed - check output above")


if __name__ == "__main__":
    main()

