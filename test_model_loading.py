#!/usr/bin/env python3

import os
import torch
import time
import gc
import psutil
from model import ModelArgs, Transformer

def get_memory_usage():
    """Get current memory usage in GB"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / 1024**3  # Convert to GB

def test_model_loading():
    print("Testing Llama 2 model loading...")
    print(f"Initial memory usage: {get_memory_usage():.2f} GB")

    # Load model args
    checkpoints_dir = "/Users/hossam.amer/Documents/workspace/Llama2_7b_weights"
    params_path = f"{checkpoints_dir}/params.json"
    model_args = ModelArgs.from_json(params_path)
    print(f"Model args loaded. Vocab size: {model_args.vocab_size}")
    print(f"Memory after loading args: {get_memory_usage():.2f} GB")

    # Set device and dtype
    device = "cpu"
    torch.set_default_dtype(torch.float16)  # Use float16 for lower memory
    print(f"Using device: {device}, dtype: {torch.get_default_dtype()}")

    # Initialize model
    print("Initializing Transformer model...")
    start_time = time.time()
    model = Transformer(model_args).to(device)
    init_time = time.time() - start_time
    print(".2f")
    print(f"Memory after model init: {get_memory_usage():.2f} GB")

    # Load checkpoint
    print("Loading checkpoint...")
    checkpoints = sorted([f for f in os.listdir(checkpoints_dir) if f.endswith(".pth")])
    checkpoint_path = os.path.join(checkpoints_dir, checkpoints[-1])
    print(f"Loading from: {checkpoint_path}")

    start_time = time.time()
    state_dict = torch.load(checkpoint_path, map_location=device, mmap=True)
    load_time = time.time() - start_time
    print(".2f")
    print(f"Memory after loading checkpoint: {get_memory_usage():.2f} GB")

    # Remap state dict
    print("Remapping state dict...")
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('feed_forward', 'ffn')
        new_state_dict[new_key] = value

    if "rope.freqs" in new_state_dict:
        del new_state_dict["rope.freqs"]

    print(f"State dict remapped. Keys: {len(new_state_dict)}")
    print(f"Memory after remapping: {get_memory_usage():.2f} GB")

    # Load state dict
    print("Loading state dict into model...")
    start_time = time.time()
    model.load_state_dict(new_state_dict, strict=True)
    state_load_time = time.time() - start_time
    print(".2f")
    print(f"Memory after loading state dict: {get_memory_usage():.2f} GB")

    # Cleanup
    del state_dict, new_state_dict
    gc.collect()
    print(f"Memory after cleanup: {get_memory_usage():.2f} GB")

    print("Model loaded successfully!")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,} ({total_params/1e9:.1f}B)")

if __name__ == "__main__":
    test_model_loading()