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


class ChatMessage(BaseModel):
    role: str
    content: Optional[str] = ""
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


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

    # Clean special tokens from response text
    cleaned_text = response_text.replace("<|im_start|>", "").replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()

    # Check if model produced tool calls in Qwen/standard format
    tool_calls = []
    content = cleaned_text

    # 1. Parse <tool_call> tags if present
    if "<tool_call>" in cleaned_text:
        parts = cleaned_text.split("<tool_call>")
        content = parts[0].strip()
        for part in parts[1:]:
            if "</tool_call>" in part:
                call_str = part.split("</tool_call>")[0].strip()
                try:
                    call_json = json.loads(call_str)
                    tool_calls.append({
                        "id": f"call_{uuid.uuid4().hex[:8]}",
                        "type": "function",
                        "function": {
                            "name": call_json.get("name"),
                            "arguments": json.dumps(call_json.get("arguments", {})) if isinstance(call_json.get("arguments"), dict) else str(call_json.get("arguments", "{}")),
                        },
                    })
                except Exception:
                    pass
    # 2. Parse direct JSON if model emitted {"name": "...", "arguments": ...}
    elif cleaned_text.startswith("{") and "name" in cleaned_text and "arguments" in cleaned_text:
        try:
            call_json = json.loads(cleaned_text)
            if "name" in call_json and "arguments" in call_json:
                tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": call_json["name"],
                        "arguments": json.dumps(call_json["arguments"]) if isinstance(call_json["arguments"], dict) else str(call_json["arguments"]),
                    },
                })
                content = ""
        except Exception:
            pass

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
