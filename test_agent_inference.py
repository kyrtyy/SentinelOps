"""
test_agent_inference.py

Tests tool-calling inference on the fine-tuned SentinelOps model (either the
merged checkpoint or an active vLLM endpoint) with a simulated production incident.

Usage (Direct local model):
    python test_agent_inference.py --model-path outputs/sentinelops-merged

Usage (Against running vLLM server):
    python test_agent_inference.py --api-base http://localhost:8000/v1
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
try:
    from common.tool_schemas import SYSTEM_PROMPT, TOOLS
except ImportError:
    from tool_schemas import SYSTEM_PROMPT, TOOLS


SAMPLE_INCIDENT = (
    "Incident detected on `checkout-service` at 2026-09-11T06:00:00Z.\n\n"
    "Raw telemetry / log excerpt:\n"
    "```\n"
    "ERROR: postgres: deadlock detected on tables `inventory` and `orders`.\n"
    "HTTP 500 InternalServerError for POST /orders/checkout.\n"
    "```\n\n"
    "Diagnose the root cause and take the appropriate remediation action."
)


def run_local_inference(model_path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading tokenizer and model from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": SAMPLE_INCIDENT},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tools=TOOLS,
        tokenize=False,
        add_generation_prompt=True,
    )

    print("\n" + "=" * 60)
    print("PROMPT SENT TO MODEL:")
    print("=" * 60)
    print(prompt)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.2,
            top_p=0.9,
            do_sample=True,
        )

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=False)
    print("\n" + "=" * 60)
    print("MODEL OUTPUT (Tool Call / Response):")
    print("=" * 60)
    print(response)


def run_vllm_api_inference(api_base, model_name="outputs/sentinelops-awq"):
    import urllib.request

    url = f"{api_base.rstrip('/')}/chat/completions"
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": SAMPLE_INCIDENT},
        ],
        "tools": TOOLS,
        "temperature": 0.2,
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    print(f"Sending request to vLLM endpoint: {url} ...")
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read().decode("utf-8"))
        print("\n" + "=" * 60)
        print("vLLM RESPONSE:")
        print("=" * 60)
        print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=None, help="Path to local model (e.g. outputs/sentinelops-merged)")
    parser.add_argument("--api-base", default=None, help="vLLM OpenAI-compatible base URL (e.g. http://localhost:8000/v1)")
    args = parser.parse_args()

    if args.api_base:
        run_vllm_api_inference(args.api_base)
    elif args.model_path:
        run_local_inference(args.model_path)
    else:
        # Default to local outputs/sentinelops-merged if exists
        default_path = "outputs/sentinelops-merged"
        if Path(default_path).exists():
            run_local_inference(default_path)
        else:
            print("Please specify --model-path or --api-base.")
            parser.print_help()


if __name__ == "__main__":
    main()
