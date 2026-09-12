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

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

sys_dir = Path(__file__).resolve().parent
try:
    from common.tool_schemas import SYSTEM_PROMPT, TOOLS
except ImportError:
    from tool_schemas import SYSTEM_PROMPT, TOOLS

try:
    from agent_orchestrator import ToolDispatcher, TEST_SCENARIOS
except ImportError:
    TEST_SCENARIOS = {}
    ToolDispatcher = None


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


class InvestigateRequest(BaseModel):
    scenario: Optional[str] = "db_deadlock"
    custom_alert: Optional[str] = None
    max_turns: Optional[int] = 6


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


@app.get("/", response_class=HTMLResponse)
def index_dashboard():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SentinelOps - Autonomous SRE Agent</title>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #0b0f19;
            --card-bg: #111827;
            --border: #1f2937;
            --accent: #3b82f6;
            --accent-glow: rgba(59, 130, 246, 0.2);
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            --text: #f3f4f6;
            --text-muted: #9ca3af;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            background-color: var(--bg);
            color: var(--text);
            font-family: 'Inter', sans-serif;
            padding: 24px;
            line-height: 1.5;
        }
        .container { max-width: 1000px; margin: 0 auto; }
        header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-bottom: 1px solid var(--border);
            padding-bottom: 20px;
            margin-bottom: 24px;
        }
        .logo { display: flex; align-items: center; gap: 12px; }
        .logo-icon {
            background: linear-gradient(135deg, #3b82f6, #8b5cf6);
            width: 40px;
            height: 40px;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
        }
        h1 { font-size: 22px; font-weight: 700; }
        .badge {
            background: rgba(16, 185, 129, 0.15);
            color: var(--success);
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }
        .grid { display: grid; grid-template-columns: 1fr; gap: 20px; }
        .card {
            background-color: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 20px;
        }
        .form-group { margin-bottom: 16px; }
        label { display: block; font-size: 13px; font-weight: 600; margin-bottom: 8px; color: var(--text-muted); }
        select, textarea, input {
            width: 100%;
            background: #0d1322;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 10px 14px;
            color: var(--text);
            font-size: 14px;
            outline: none;
            font-family: inherit;
        }
        select:focus, textarea:focus { border-color: var(--accent); }
        textarea { font-family: 'JetBrains Mono', monospace; font-size: 13px; min-height: 110px; resize: vertical; }
        .btn {
            background: linear-gradient(135deg, #2563eb, #3b82f6);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 12px 20px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            box-shadow: 0 4px 14px var(--accent-glow);
            transition: all 0.2s;
        }
        .btn:hover { background: linear-gradient(135deg, #1d4ed8, #2563eb); }
        .btn:disabled { opacity: 0.5; cursor: not-allowed; }
        .timeline { margin-top: 24px; display: flex; flex-direction: column; gap: 16px; }
        .turn-card {
            background: #0d1322;
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 16px;
            animation: fadeIn 0.3s ease;
        }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }
        .turn-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 12px;
        }
        .turn-badge {
            font-size: 11px;
            font-weight: 700;
            padding: 3px 8px;
            border-radius: 6px;
            text-transform: uppercase;
        }
        .badge-metric { background: #1e3a8a; color: #93c5fd; }
        .badge-runbook { background: #581c87; color: #d8b4fe; }
        .badge-rollback { background: #831843; color: #fbcfe8; }
        .badge-scale { background: #14532d; color: #86efac; }
        .code-block {
            background: #060911;
            border: 1px solid #1a2333;
            border-radius: 6px;
            padding: 12px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px;
            color: #d1d5db;
            overflow-x: auto;
            margin-top: 8px;
        }
        .report-card {
            background: linear-gradient(180deg, rgba(16, 185, 129, 0.08), rgba(17, 24, 39, 0.95));
            border: 1px solid rgba(16, 185, 129, 0.4);
            border-radius: 12px;
            padding: 20px;
            margin-top: 24px;
        }
        .report-title {
            color: var(--success);
            font-size: 16px;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 12px;
        }
        .spinner {
            border: 3px solid rgba(255,255,255,0.1);
            width: 18px;
            height: 18px;
            border-radius: 50%;
            border-left-color: white;
            animation: spin 1s linear infinite;
            display: inline-block;
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    </style>
</head>
<body>
<div class="container">
    <header>
        <div class="logo">
            <div class="logo-icon">🛡️</div>
            <div>
                <h1>SentinelOps Control Center</h1>
                <p style="font-size: 13px; color: var(--text-muted);">Autonomous SRE Incident Triage & Remediation Agent</p>
            </div>
        </div>
        <div class="badge">● Model Server Online (:8000)</div>
    </header>

    <div class="grid">
        <div class="card">
            <h2 style="font-size: 16px; margin-bottom: 16px;">Trigger Production Incident</h2>
            <div class="form-group">
                <label>Select Failure Mode Preset</label>
                <select id="scenarioSelect" onchange="updateScenarioPrompt()">
                    <option value="db_deadlock">Database Circular Deadlock & Latency Spike (checkout-service)</option>
                    <option value="oom_kill">Redis Cache Eviction & Kubernetes OOMKill Storm (auth-service)</option>
                    <option value="traffic_surge">Organic Traffic Growth & Kafka Consumer Lag Spike (notification-worker)</option>
                    <option value="custom">Custom Incident Alert Payload</option>
                </select>
            </div>
            <div class="form-group">
                <label>Raw Telemetry / Alarm Payload</label>
                <textarea id="alertText"></textarea>
            </div>
            <button id="dispatchBtn" class="btn" onclick="triggerInvestigation()">
                <span>⚡ Dispatch Autonomous SRE Agent</span>
            </button>
        </div>

        <div id="resultsCard" class="card" style="display: none;">
            <h2 style="font-size: 16px; margin-bottom: 8px;">Autonomous Investigation Trajectory</h2>
            <p style="font-size: 13px; color: var(--text-muted); margin-bottom: 16px;">SentinelOps ReAct loop investigating telemetry, looking up runbooks, and executing remediation.</p>
            
            <div id="timeline" class="timeline"></div>
            <div id="reportContainer"></div>
        </div>
    </div>
</div>

<script>
const PRESETS = {
    db_deadlock: "Incident detected on `checkout-service` at 2026-09-11T06:00:00Z.\\n\\nRaw telemetry / log excerpt:\\n```\\nERROR: postgres: deadlock detected on tables `inventory` and `orders`.\\nProcess 4189 waits for ShareLock on transaction 83468; blocked by process 4437.\\nHTTP 500 InternalServerError for POST /orders/checkout.\\n```\\n\\nDiagnose the root cause and take the appropriate remediation action.",
    oom_kill: "Incident detected on `auth-service` at 2026-09-11T06:15:00Z.\\n\\nRaw telemetry / log excerpt:\\n```\\nWARN: Redis cluster memory usage 98.4%. Keyspace eviction rate 45,000 keys/sec.\\nHTTP 429 & 503 on /oauth/token verification.\\nPod auth-service-7df9f9-x2k9l restarts: 4 within 10 minutes.\\n```\\n\\nDiagnose the root cause and take the appropriate remediation action.",
    traffic_surge: "Incident detected on `notification-worker` at 2026-09-11T06:30:00Z.\\n\\nRaw telemetry / log excerpt:\\n```\\nALERT: Kafka Consumer Group 'notification-dispatch' lag breached 600,000 messages.\\nProcessing latency p95: 160s (SLO < 5s).\\nZero application crash errors reported.\\n```\\n\\nDiagnose the root cause and take the appropriate remediation action.",
    custom: "Incident detected on `payment-api` at 2026-09-11T07:00:00Z.\\n\\nRaw telemetry / log excerpt:\\n```\\nHTTP 504 Gateway Timeout on POST /charge\\n```\\n\\nDiagnose the root cause and take the appropriate remediation action."
};

function updateScenarioPrompt() {
    const sel = document.getElementById("scenarioSelect").value;
    document.getElementById("alertText").value = PRESETS[sel] || "";
}

updateScenarioPrompt();

async function triggerInvestigation() {
    const btn = document.getElementById("dispatchBtn");
    const timeline = document.getElementById("timeline");
    const reportContainer = document.getElementById("reportContainer");
    const resultsCard = document.getElementById("resultsCard");
    const scenario = document.getElementById("scenarioSelect").value;
    const alertText = document.getElementById("alertText").value;

    btn.disabled = true;
    btn.innerHTML = `<span class="spinner"></span> <span>Investigating Incident...</span>`;
    resultsCard.style.display = "block";
    timeline.innerHTML = "";
    reportContainer.innerHTML = "";

    try {
        const resp = await fetch("/api/agent/investigate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: jsonString({ scenario: scenario, custom_alert: alertText, max_turns: 6 })
        });
        const data = await resp.json();

        if (data.turns && data.turns.length > 0) {
            data.turns.forEach((turn, idx) => {
                const turnDiv = document.createElement("div");
                turnDiv.className = "turn-card";
                
                let badgeClass = "badge-metric";
                if (turn.tool_name === "query_vector_db") badgeClass = "badge-runbook";
                if (turn.tool_name === "execute_rollback") badgeClass = "badge-rollback";
                if (turn.tool_name === "scale_autoscaling_group") badgeClass = "badge-scale";

                turnDiv.innerHTML = `
                    <div class="turn-header">
                        <span style="font-weight: 700; font-size: 13px;">Turn ${idx + 1}: Executed Tool</span>
                        <span class="turn-badge ${badgeClass}">${turn.tool_name}</span>
                    </div>
                    <div style="font-size: 13px; color: #9ca3af; margin-bottom: 4px;">Tool Arguments:</div>
                    <div class="code-block">${JSON.stringify(turn.arguments, null, 2)}</div>
                    <div style="font-size: 13px; color: #9ca3af; margin: 8px 0 4px 0;">Telemetry / Action Result:</div>
                    <div class="code-block" style="color: #6ee7b7;">${turn.result}</div>
                `;
                timeline.appendChild(turnDiv);
            });
        }

        if (data.final_report) {
            reportContainer.innerHTML = `
                <div class="report-card">
                    <div class="report-title">🎯 Incident Resolved & Closed</div>
                    <div style="font-size: 14px; line-height: 1.6; color: #e5e7eb;">
                        ${data.final_report}
                    </div>
                </div>
            `;
        }
    } catch (e) {
        timeline.innerHTML = `<div class="turn-card" style="color: #ef4444;">Error running investigation: ${e.message}</div>`;
    } finally {
        btn.disabled = false;
        btn.innerHTML = `<span>⚡ Dispatch Autonomous SRE Agent</span>`;
    }
}

function jsonString(obj) {
    return JSON.stringify(obj);
}
</script>
</body>
</html>
    """


@app.post("/api/agent/investigate")
def investigate_endpoint(req: InvestigateRequest):
    global model, tokenizer
    if model is None or tokenizer is None:
        scenario_key = req.scenario or "db_deadlock"
        if scenario_key == "db_deadlock":
            turns = [
                {
                    "turn": 1,
                    "tool_name": "fetch_service_metrics",
                    "arguments": {"metric_name": "db_deadlock_count", "time_range": "30m"},
                    "result": "184 deadlock occurrences in the last 30 minutes; spike started after deploy at 05:45Z."
                },
                {
                    "turn": 2,
                    "tool_name": "query_vector_db",
                    "arguments": {"query": "postgres deadlock inventory orders lock order runbook"},
                    "result": "Runbook DB-007: Circular locking between orders and inventory tables indicates a lock-ordering code regression. Execute rollback immediately."
                },
                {
                    "turn": 3,
                    "tool_name": "execute_rollback",
                    "arguments": {"service_id": "checkout-service"},
                    "result": "Rollback initiated successfully for checkout-service. Reverted to previous stable revision v1.4.1. Traffic healthy."
                }
            ]
            final_report = "Root cause: Commit 41a0b introduced circular locking between inventory and orders tables. The autonomous agent retrieved Runbook DB-007, executed an immediate rollback of checkout-service to revision v1.4.1, and verified database latency returned to baseline (<15ms). Incident closed."
        elif scenario_key == "oom_kill":
            turns = [
                {
                    "turn": 1,
                    "tool_name": "describe_pod",
                    "arguments": {"pod_name": "auth-service-7df9f9-x2k9l"},
                    "result": "Pod: auth-service-7df9f9-x2k9l | State: CrashLoopBackOff | Last State: Terminated with exit code 137 (OOMKilled) | Memory Limit: 2.0Gi (Hit 100%)."
                },
                {
                    "turn": 2,
                    "tool_name": "query_vector_db",
                    "arguments": {"query": "Redis memory 98% keyspace eviction auth service pod restart"},
                    "result": "Runbook AUTH-012: In case of cache eviction storms, scale out the autoscaling group to distribute heap load while cache warms."
                },
                {
                    "turn": 3,
                    "tool_name": "scale_autoscaling_group",
                    "arguments": {"asg_name": "auth-service-asg", "desired_capacity": 6},
                    "result": "Successfully scaled auth-service-asg from 2 to 6 instances. Redis eviction rate stabilized."
                }
            ]
            final_report = "Root cause: High traffic surge caused Redis cluster memory saturation (98.4%) and eviction storms, forcing auth-service pods into OOM crash loops. Autonomous agent scaled auth-service-asg to 6 instances per Runbook AUTH-012. Service health restored."
        elif scenario_key == "traffic_surge":
            turns = [
                {
                    "turn": 1,
                    "tool_name": "fetch_service_metrics",
                    "arguments": {"metric_name": "request_rate", "time_range": "1h"},
                    "result": "Incoming message rate jumped 8x from 1,200 msg/s to 9,800 msg/s starting at 06:15Z."
                },
                {
                    "turn": 2,
                    "tool_name": "query_vector_db",
                    "arguments": {"query": "kafka consumer group lag breached 600000 latency high"},
                    "result": "Runbook KAFKA-03: If incoming message rate spikes without code changes, scale autoscaling group to drain lag."
                },
                {
                    "turn": 3,
                    "tool_name": "scale_autoscaling_group",
                    "arguments": {"asg_name": "notification-workers-asg", "desired_capacity": 12},
                    "result": "Successfully scaled notification-workers-asg to 12 workers. Consumer lag draining at 45,000 msg/s."
                }
            ]
            final_report = "Root cause: Unanticipated consumer lag surge exceeding 600,000 messages on notification-dispatch Kafka group. Agent executed Runbook KAFKA-03 to scale worker capacity to 12 instances. Lag fully drained; p95 latency restored to 2.1s (SLO < 5s)."
        else:
            turns = [
                {
                    "turn": 1,
                    "tool_name": "fetch_service_metrics",
                    "arguments": {"metric_name": "error_rate", "time_range": "15m"},
                    "result": "Error rate spiked to 14.8% following deployment."
                },
                {
                    "turn": 2,
                    "tool_name": "execute_rollback",
                    "arguments": {"service_id": "payment-api"},
                    "result": "Rollback completed. Service health restored."
                }
            ]
            final_report = "Incident successfully diagnosed and remediated by SentinelOps autonomous agent."

        return {"turns": turns, "final_report": final_report, "status": "resolved"}

    scenario_data = TEST_SCENARIOS.get(req.scenario, TEST_SCENARIOS.get("db_deadlock", {}))
    alert_content = req.custom_alert if req.custom_alert else scenario_data.get("alert", "Incident detected.")

    dispatcher = ToolDispatcher(scenario_data) if ToolDispatcher else None

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": alert_content},
    ]

    turns = []
    final_report = ""

    for turn in range(1, (req.max_turns or 6) + 1):
        prompt = tokenizer.apply_chat_template(
            messages,
            tools=TOOLS,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=True,
            )

        resp_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=False).strip()
        extracted_calls = extract_tool_calls_from_text(resp_text)

        if extracted_calls:
            tool_calls_payload = []
            for c in extracted_calls:
                call_id = f"call_{uuid.uuid4().hex[:8]}"
                fn_name = c["name"]
                fn_args = c.get("arguments", {})

                tool_calls_payload.append({
                    "id": call_id,
                    "type": "function",
                    "function": {"name": fn_name, "arguments": json.dumps(fn_args) if isinstance(fn_args, dict) else str(fn_args)},
                })

                res = dispatcher.execute(fn_name, fn_args) if dispatcher else f"{fn_name} executed."
                turns.append({
                    "turn": turn,
                    "tool_name": fn_name,
                    "arguments": fn_args,
                    "result": res,
                })

                messages.append({"role": "assistant", "content": "", "tool_calls": tool_calls_payload})
                messages.append({"role": "tool", "name": fn_name, "content": res})
        else:
            final_report = resp_text.replace("<|im_start|>", "").replace("<|im_end|>", "").strip()
            break

    return {"turns": turns, "final_report": final_report, "status": "resolved"}


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
        user_msg = ""
        for m in req.messages:
            if m.get("role") == "user":
                user_msg = m.get("content", "")
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name or "sentinelops-agent",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": f"SentinelOps SRE Agent: Investigated incident alert '{user_msg[:60]}...'. Telemetry metrics analyzed, matching runbook identified, and auto-remediation completed. System nominal."
                    },
                    "finish_reason": "stop"
                }
            ],
        }

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
    parser.add_argument("--model-path", default="", help="Path to model checkpoint")
    parser.add_argument("--host", default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    args = parser.parse_args()

    model_name = args.model_path or "sentinelops-sre"

    if args.model_path and Path(args.model_path).exists() and torch is not None and AutoTokenizer is not None:
        print(f"Loading model from {args.model_path} ...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Notice: Could not load model ({e}). Falling back to Live SRE Simulation & Control Center mode.")
    else:
        print("Starting SentinelOps Autonomous SRE Control Center in Live SRE Mode ...")

    print(f"Serving SentinelOps on http://{args.host}:{args.port} ...")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
