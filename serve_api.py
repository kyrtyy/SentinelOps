"""
serve_api.py

Lightweight, high-performance OpenAI-compatible FastAPI server for SentinelOps.
Can serve both AWQ INT4 quantized models and unquantized bf16 checkpoints
directly using PyTorch / AutoAWQ / Transformers with zero driver/CUDA conflict.

Usage:
    python serve_api.py --model-path outputs/sentinelops-awq --port 8000
    python serve_api.py --model-path outputs/sentinelops-merged --port 8000
"""
import argparse
import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys_dir = Path(__file__).resolve().parent
try:
    from common.tool_schemas import TOOLS
except ImportError:
    from tool_schemas import TOOLS


app = FastAPI(title="SentinelOps Model Server")
model = None
tokenizer = None
model_name = ""


def extract_tool_calls_from_text(text: str) -> List[Dict[str, Any]]:
    if not text:
        return []
    cleaned = text.replace("<|im_start|>", "").replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
    results = []
    start_idx = -1
    brace_count = 0
    in_string = False
    escape = False

    for i, ch in enumerate(cleaned):
        if ch == '"' and not escape:
            in_string = not in_string
        elif ch == '\\' and in_string:
            escape = not escape
            continue
        elif not in_string:
            if ch == '{':
                if brace_count == 0:
                    start_idx = i
                brace_count += 1
            elif ch == '}':
                if brace_count > 0:
                    brace_count -= 1
                    if brace_count == 0 and start_idx != -1:
                        candidate = cleaned[start_idx:i+1]
                        try:
                            obj = json.loads(candidate)
                            if isinstance(obj, dict) and "name" in obj:
                                results.append(obj)
                        except Exception:
                            pass
                        start_idx = -1
        escape = False
    return results


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "sentinelops"
    messages: List[Dict[str, Any]]
    tools: Optional[List[Dict[str, Any]]] = None
    temperature: Optional[float] = 0.2
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = 512


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": model_name, "object": "model", "created": int(time.time()), "owned_by": "sentinelops"}],
    }


@app.post("/v1/chat/completions")
def create_chat_completion(req: ChatCompletionRequest):
    global model, tokenizer
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Model is not loaded")

    tools_to_use = req.tools if req.tools is not None else TOOLS

    try:
        prompt = tokenizer.apply_chat_template(
            req.messages,
            tools=tools_to_use,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error applying chat template: {e}")

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=req.max_tokens or 512,
            temperature=max(req.temperature or 0.2, 0.01),
            top_p=req.top_p or 0.9,
            do_sample=True,
        )

    response_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=False).strip()

    cleaned_text = response_text.replace("<|im_start|>", "").replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
    extracted_calls = extract_tool_calls_from_text(response_text)

    tool_calls = []
    content = ""

    if extracted_calls:
        for c in extracted_calls:
            tool_calls.append({
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": c.get("name"),
                    "arguments": json.dumps(c.get("arguments", {})) if isinstance(c.get("arguments"), dict) else str(c.get("arguments", "{}")),
                },
            })
    else:
        content = cleaned_text

    message_obj = {"role": "assistant", "content": content}
    if tool_calls:
        message_obj["tool_calls"] = tool_calls

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": message_obj,
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }
        ],
    }


def main():
    global model, tokenizer, model_name
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="outputs/sentinelops-awq", help="Path to model checkpoint")
    parser.add_argument("--host", default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    args = parser.parse_args()

    model_name = args.model_path
    print(f"Loading model from {args.model_path} ...")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Check if AWQ quantized
    is_awq = False
    config_file = Path(args.model_path) / "config.json"
    if config_file.exists():
        with open(config_file) as f:
            cfg = json.load(f)
            if "quantization_config" in cfg:
                is_awq = True

    if is_awq:
        try:
            from awq import AutoAWQForCausalLM
            print("Detected AWQ quantized model. Loading with AutoAWQ...")
            model = AutoAWQForCausalLM.from_quantized(
                args.model_path,
                fuse_layers=True,
                safetensors=True,
            )
        except ImportError:
            print("AutoAWQ not installed. Loading via Transformers...")
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
    else:
        print("Loading standard model via Transformers...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    print(f"Serving SentinelOps on http://{args.host}:{args.port}/v1 ...")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
