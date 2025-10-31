#!/usr/bin/env python3
"""
Benchmark original vs REAP pruned Qwen3-Next models
Generates comparison charts and metrics
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

MODELS = {
    "Original (100%)": "Qwen/Qwen3-Next-80B-A3B-Instruct",
    "REAP 50%": "kevint00/Qwen3-Next-80B-A3B-Instruct-REAP-50pct",
    "REAP 30%": "kevint00/Qwen3-Next-80B-A3B-Instruct-REAP-30pct",
    "REAP 10%": "kevint00/Qwen3-Next-80B-A3B-Instruct-REAP-10pct",
    "REAP 5%": "kevint00/Qwen3-Next-80B-A3B-Instruct-REAP-5pct",
    "REAP 2%": "kevint00/Qwen3-Next-80B-A3B-Instruct-REAP-2pct",
}

BENCHMARK_PROMPTS = [
    "Write a Python function to implement binary search:\n\n```python\n",
    "Explain how neural networks work:\n\n",
    "def merge_sort(arr):\n    # Complete this sorting algorithm\n",
    "What is the capital of France? Answer:",
    "Translate to Spanish: Hello, how are you?\n\nSpanish:",
]


def benchmark_model(name, model_id):
    """Benchmark a single model"""
    print(f"\n{'='*70}")
    print(f"Benchmarking: {name}")
    print(f"{'='*70}")
    
    results = {
        "name": name,
        "model_id": model_id,
        "load_time": 0,
        "params_b": 0,
        "expert_count": 0,
        "avg_tokens_per_sec": 0,
        "avg_latency": 0,
        "memory_gb": 0,
        "success": False
    }
    
    try:
        # Load model
        print("⏳ Loading model...")
        start = time.time()
        
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        
        results["load_time"] = time.time() - start
        results["params_b"] = sum(p.numel() for p in model.parameters()) / 1e9
        
        print(f"✅ Loaded in {results['load_time']:.1f}s")
        print(f"✅ Parameters: {results['params_b']:.2f}B")
        
        # Count experts
        for module in model.modules():
            if hasattr(module, 'experts') and hasattr(module.experts, '__len__'):
                results["expert_count"] += len(module.experts)
        
        if results["expert_count"] > 0:
            print(f"✅ Total experts: {results['expert_count']}")
        
        # Measure memory
        torch.cuda.synchronize()
        results["memory_gb"] = torch.cuda.max_memory_allocated() / 1e9
        print(f"✅ GPU Memory: {results['memory_gb']:.2f}GB")
        
        # Run inference benchmarks
        print("\n🧪 Running inference benchmarks...")
        
        latencies = []
        token_speeds = []
        
        for i, prompt in enumerate(BENCHMARK_PROMPTS, 1):
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            # Warmup
            if i == 1:
                with torch.no_grad():
                    model.generate(**inputs, max_new_tokens=10, do_sample=False)
            
            # Benchmark
            torch.cuda.synchronize()
            start = time.time()
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            torch.cuda.synchronize()
            gen_time = time.time() - start
            
            tokens = len(outputs[0]) - len(inputs.input_ids[0])
            tps = tokens / gen_time
            
            latencies.append(gen_time)
            token_speeds.append(tps)
            
            print(f"  Prompt {i}: {tps:.1f} tok/s ({gen_time:.2f}s)")
        
        results["avg_tokens_per_sec"] = sum(token_speeds) / len(token_speeds)
        results["avg_latency"] = sum(latencies) / len(latencies)
        results["success"] = True
        
        print(f"\n✅ {name} - Benchmark complete")
        print(f"   Avg throughput: {results['avg_tokens_per_sec']:.1f} tok/s")
        print(f"   Avg latency: {results['avg_latency']:.2f}s")
        
        # Cleanup
        del model, tokenizer
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
    except Exception as e:
        print(f"\n❌ {name} - Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
    
    return results


def create_charts(results_df, output_dir="benchmark_results"):
    """Create comparison charts"""
    Path(output_dir).mkdir(exist_ok=True)
    
    # Filter successful results
    df = results_df[results_df['success'] == True].copy()
    
    if len(df) == 0:
        print("⚠️  No successful benchmarks to chart")
        return
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Chart 1: Throughput comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(df['name'], df['avg_tokens_per_sec'], color='steelblue', alpha=0.8)
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Throughput (tokens/sec)', fontsize=12, fontweight='bold')
    ax.set_title('REAP Pruning: Throughput Comparison', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/throughput_comparison.png', dpi=300)
    print(f"✅ Saved: {output_dir}/throughput_comparison.png")
    
    # Chart 2: Memory vs Parameters
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    x = range(len(df))
    ax1.bar([i - 0.2 for i in x], df['params_b'], 0.4, label='Parameters (B)', color='coral', alpha=0.8)
    ax1.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Parameters (Billions)', fontsize=12, fontweight='bold', color='coral')
    ax1.tick_params(axis='y', labelcolor='coral')
    
    ax2 = ax1.twinx()
    ax2.bar([i + 0.2 for i in x], df['memory_gb'], 0.4, label='GPU Memory (GB)', color='steelblue', alpha=0.8)
    ax2.set_ylabel('GPU Memory (GB)', fontsize=12, fontweight='bold', color='steelblue')
    ax2.tick_params(axis='y', labelcolor='steelblue')
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['name'], rotation=45, ha='right')
    ax1.set_title('Model Size & Memory Usage', fontsize=14, fontweight='bold')
    
    fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
    plt.tight_layout()
    plt.savefig(f'{output_dir}/size_memory_comparison.png', dpi=300)
    print(f"✅ Saved: {output_dir}/size_memory_comparison.png")
    
    # Chart 3: Speedup vs Compression
    if 'Original (100%)' in df['name'].values:
        orig_idx = df[df['name'] == 'Original (100%)'].index[0]
        orig_tps = df.loc[orig_idx, 'avg_tokens_per_sec']
        orig_params = df.loc[orig_idx, 'params_b']
        
        pruned_df = df[df['name'] != 'Original (100%)'].copy()
        pruned_df['speedup'] = pruned_df['avg_tokens_per_sec'] / orig_tps
        pruned_df['compression'] = pruned_df['params_b'] / orig_params * 100
        
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(pruned_df['compression'], pruned_df['speedup'], 
                           s=200, c=pruned_df['avg_tokens_per_sec'], 
                           cmap='viridis', alpha=0.7, edgecolors='black', linewidth=2)
        
        for idx, row in pruned_df.iterrows():
            ax.annotate(row['name'], (row['compression'], row['speedup']),
                       xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax.axhline(y=1.0, color='red', linestyle='--', label='Original speed', linewidth=2)
        ax.set_xlabel('Model Size (% of original)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Speedup vs Original', fontsize=12, fontweight='bold')
        ax.set_title('REAP Pruning: Speedup vs Compression', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Throughput (tok/s)', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/speedup_vs_compression.png', dpi=300)
        print(f"✅ Saved: {output_dir}/speedup_vs_compression.png")
    
    plt.close('all')


def main():
    print("\n" + "="*70)
    print("REAP BENCHMARK: Original vs Pruned Models")
    print("="*70)
    
    # Run benchmarks
    all_results = []
    
    for name, model_id in MODELS.items():
        result = benchmark_model(name, model_id)
        all_results.append(result)
        
        # Save intermediate results
        df = pd.DataFrame(all_results)
        df.to_csv('benchmark_results/intermediate_results.csv', index=False)
        
        time.sleep(2)  # Cool down between models
    
    # Create DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save results
    Path("benchmark_results").mkdir(exist_ok=True)
    results_df.to_csv('benchmark_results/full_results.csv', index=False)
    results_df.to_json('benchmark_results/full_results.json', orient='records', indent=2)
    
    # Create charts
    create_charts(results_df)
    
    # Print summary table
    print("\n" + "="*70)
    print("BENCHMARK SUMMARY")
    print("="*70)
    
    print(results_df[['name', 'params_b', 'expert_count', 'avg_tokens_per_sec', 
                      'avg_latency', 'memory_gb', 'success']].to_string(index=False))
    
    print(f"\n✅ Results saved to: benchmark_results/")
    print(f"   - full_results.csv")
    print(f"   - full_results.json")
    print(f"   - throughput_comparison.png")
    print(f"   - size_memory_comparison.png")
    print(f"   - speedup_vs_compression.png")


if __name__ == "__main__":
    main()

