"""
merge_lora.py

Merges a trained LoRA adapter into its base model and saves a single,
self-contained bf16 checkpoint - the artifact the next phase (AWQ
quantization + vLLM serving) consumes.

Usage:
    python merge_lora.py \
        --base-model Qwen/Qwen2.5-Coder-7B-Instruct \
        --adapter outputs/sentinelops-lora \
        --output outputs/sentinelops-merged
"""
import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True, help="Path to the saved LoRA adapter (train_ddp.py's output_dir)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    print(f"Loading base model {args.base_model} ...")
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(args.adapter)

    print(f"Loading LoRA adapter from {args.adapter} ...")
    model = PeftModel.from_pretrained(base_model, args.adapter)

    print("Merging adapter into base weights ...")
    model = model.merge_and_unload()

    print(f"Saving merged model to {args.output} ...")
    model.save_pretrained(args.output, safe_serialization=True)
    tokenizer.save_pretrained(args.output)
    print("Done. This merged checkpoint is the input to quantize_awq.py in the next phase.")


if __name__ == "__main__":
    main()
