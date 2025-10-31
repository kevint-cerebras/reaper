#!/usr/bin/env python3
"""
Interactive chat with REAP pruned Qwen3-Next model
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys

def main():
    # Configuration
    model_id = "kevint00/Qwen3-Next-80B-A3B-Instruct-REAP-50pct"
    cache_dir = "/tmp/huggingface"
    
    print("="*70)
    print("REAP PRUNED MODEL CHAT")
    print("="*70)
    print(f"\nLoading model: {model_id}")
    print("This may take a few minutes...\n")
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        cache_dir=cache_dir
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        cache_dir=cache_dir
    )
    
    print("✅ Model loaded!\n")
    print("="*70)
    print("Chat Instructions:")
    print("  - Type your message and press Enter")
    print("  - Type 'quit', 'exit', or Ctrl+C to exit")
    print("  - Type 'clear' to reset conversation history")
    print("="*70)
    
    # Chat loop
    conversation_history = []
    
    while True:
        try:
            # Get user input
            user_input = input("\n👤 You: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if user_input.lower() == 'clear':
                conversation_history = []
                print("\n🗑️  Conversation history cleared")
                continue
            
            # Build prompt with history
            if conversation_history:
                prompt = ""
                for role, text in conversation_history:
                    if role == "user":
                        prompt += f"User: {text}\n"
                    else:
                        prompt += f"Assistant: {text}\n"
                prompt += f"User: {user_input}\nAssistant:"
            else:
                prompt = f"User: {user_input}\nAssistant:"
            
            # Generate response
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            print("\n🤖 Assistant: ", end="", flush=True)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,  # Greedy decoding
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            
            # Decode response
            full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the assistant's response
            assistant_response = full_response[len(prompt):].strip()
            
            # Stop at newline for cleaner responses
            if "\nUser:" in assistant_response:
                assistant_response = assistant_response.split("\nUser:")[0].strip()
            
            print(assistant_response)
            
            # Update conversation history
            conversation_history.append(("user", user_input))
            conversation_history.append(("assistant", assistant_response))
            
            # Keep only last 10 exchanges to prevent context overflow
            if len(conversation_history) > 20:
                conversation_history = conversation_history[-20:]
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()

