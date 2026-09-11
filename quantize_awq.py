"""
quantize_awq.py

Quantizes the merged SentinelOps 7B model checkpoint to AWQ INT4 (Activation-aware
Weight Quantization) for high-throughput, low-latency vLLM serving on single-GPU
instances (e.g., AWS EC2 g5.xlarge / NVIDIA A10G 24GB or RTX 3090/4090).

Prerequisites:
    pip install autoawq

Usage:
    python quantize_awq.py \
        --model-path outputs/sentinelops-merged \
        --output-path outputs/sentinelops-awq
"""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="outputs/sentinelops-merged", help="Path to merged bf16 model")
    parser.add_argument("--output-path", default="outputs/sentinelops-awq", help="Output path for quantized model")
    parser.add_argument("--w-bit", type=int, default=4, help="Weight bit width (default: 4)")
    parser.add_argument("--q-group-size", type=int, default=128, help="Quantization group size (default: 128)")
    parser.add_argument("--version", default="GEMM", choices=["GEMM", "GEMV", "MARLIN"], help="Kernel version")
    args = parser.parse_args()

    try:
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer
    except ImportError:
        print("Error: autoawq is required for AWQ quantization.")
        print("Install it with: pip install autoawq")
        return

    print(f"Loading merged model from {args.model_path} ...")
    model = AutoAWQForCausalLM.from_pretrained(args.model_path, safetensors=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    quant_config = {
        "zero_point": True,
        "q_group_size": args.q_group_size,
        "w_bit": args.w_bit,
        "version": args.version,
    }

    print(f"Quantizing model with config: {quant_config} ...")
    model.quantize(tokenizer, quant_config=quant_config)

    print(f"Saving quantized model to {args.output_path} ...")
    Path(args.output_path).mkdir(parents=True, exist_ok=True)
    model.save_quantized(args.output_path)
    tokenizer.save_pretrained(args.output_path)

    print(f"Successfully saved AWQ INT4 model to {args.output_path}")
    print("\nTo serve with vLLM:")
    print(f"    vllm serve {args.output_path} --quantization awq --port 8000")


if __name__ == "__main__":
    main()
