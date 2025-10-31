#!/usr/bin/env python3
"""Upload all REAP pruned models to HuggingFace"""

from huggingface_hub import HfApi
import os
from pathlib import Path

api = HfApi(token=os.environ["HF_TOKEN"])

# Map directory names to HF repo names
models = {
    "/workspace/reaper/artifacts/Qwen3-Next-80B-A3B-Instruct/evol-codealpaca-v1/pruned_models/reap-seed_42-0.50": 
        "kevint00/Qwen3-Next-80B-A3B-Instruct-REAP-50pct",
    "/workspace/reaper/artifacts/Qwen3-Next-80B-A3B-Instruct/evol-codealpaca-v1/pruned_models/reap-seed_42-0.70": 
        "kevint00/Qwen3-Next-80B-A3B-Instruct-REAP-30pct",
    "/workspace/reaper/artifacts/Qwen3-Next-80B-A3B-Instruct/evol-codealpaca-v1/pruned_models/reap-seed_42-0.90": 
        "kevint00/Qwen3-Next-80B-A3B-Instruct-REAP-10pct",
    "/workspace/reaper/artifacts/Qwen3-Next-80B-A3B-Instruct/evol-codealpaca-v1/pruned_models/reap-seed_42-0.95": 
        "kevint00/Qwen3-Next-80B-A3B-Instruct-REAP-5pct",
    "/workspace/reaper/artifacts/Qwen3-Next-80B-A3B-Instruct/evol-codealpaca-v1/pruned_models/reap-seed_42-0.98": 
        "kevint00/Qwen3-Next-80B-A3B-Instruct-REAP-2pct",
}

for model_path, repo_id in models.items():
    if not Path(model_path).exists():
        print(f"⚠️  Skipping {repo_id} - path not found")
        continue
    
    print(f"\n{'='*60}")
    print(f"Uploading {repo_id}")
    print(f"From: {model_path}")
    print(f"{'='*60}")
    
    try:
        # Create repo
        api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
        
        # Upload folder
        api.upload_folder(
            folder_path=model_path,
            repo_id=repo_id,
            repo_type="model",
        )
        
        print(f"✅ Uploaded {repo_id}")
    except Exception as e:
        print(f"❌ Failed {repo_id}: {e}")

print("\n" + "="*60)
print("🎉 Upload complete!")
print("="*60)

