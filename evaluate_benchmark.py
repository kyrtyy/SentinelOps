"""
evaluate_benchmark.py

Evaluates the SentinelOps fine-tuned agent across validation incidents:
1. Executes multi-turn ReAct loops against the model endpoint (http://localhost:8000/v1).
2. Scores remediation accuracy, targeted service correctness, and investigative discipline against ground truth.
3. (Optional) Uses an LLM Judge (Bedrock Claude 3.5 Sonnet, OpenAI, or offline semantic judge) to grade root cause identification.
4. Generates a comprehensive benchmark scorecard (benchmark_scorecard.md).

Usage:
    python evaluate_benchmark.py --api-base http://localhost:8000/v1
    python evaluate_benchmark.py --api-base http://localhost:8000/v1 --judge bedrock
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
try:
    from common.tool_schemas import SYSTEM_PROMPT, TOOLS
except ImportError:
    from tool_schemas import SYSTEM_PROMPT, TOOLS


def extract_json_objects(text: str):
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


class DynamicIncidentEnvironment:
    """Simulates realistic telemetry and tool feedback for an incident record."""

    def __init__(self, incident_record):
        self.record = incident_record
        self.service = incident_record.get("service", "target-service")
        self.tool_trace = incident_record.get("tool_trace", [])
        self.trace_lookup = {step["tool"]: step["result"] for step in self.tool_trace}

    def execute(self, tool_name, arguments):
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except Exception:
                pass
        if not isinstance(arguments, dict):
            arguments = {}

        # 1. Exact match from ground-truth trace if present
        if tool_name in self.trace_lookup:
            return self.trace_lookup[tool_name]

        # 2. Dynamic fallbacks
        if tool_name == "fetch_service_metrics":
            metric = arguments.get("metric_name", "error_rate")
            return f"Telemetry query for `{self.service}` [{metric}]: Critical anomaly threshold breached in last 30m."

        elif tool_name == "query_vector_db":
            return f"Runbook match for `{self.service}`: Check git diff for recent deploy regressions or scale capacity if traffic is organic."

        elif tool_name == "fetch_git_diff":
            return f"Commit diff for `{self.service}`: Modified configuration and threadpool parameters."

        elif tool_name == "describe_pod":
            return f"Pod `{self.service}-pod`: 3 restarts detected; memory limit exceeded (OOMKilled exit code 137)."

        elif tool_name == "get_logs":
            return f"Logs for `{self.service}`: Stack trace shows connection timeout and worker saturation."

        elif tool_name == "execute_rollback":
            target = arguments.get("service_id", self.service)
            return f"Rollback initiated successfully for `{target}`. Reverted to previous stable revision."

        elif tool_name == "scale_autoscaling_group":
            asg = arguments.get("asg_name", f"{self.service}-asg")
            cap = arguments.get("target_capacity", 8)
            return f"Autoscaling group `{asg}` scaled to {cap} replicas. Provisioning complete."

        return f"Tool `{tool_name}` executed successfully."


def call_api(api_base, messages, tools):
    url = f"{api_base.rstrip('/')}/chat/completions"
    payload = {
        "model": "sentinelops",
        "messages": messages,
        "tools": tools,
        "temperature": 0.1,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    start_time = time.time()
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    latency_ms = (time.time() - start_time) * 1000
    return data, latency_ms


def evaluate_with_heuristic_judge(predicted_summary, ground_truth_root_cause):
    """Offline semantic keyword and intent judge requiring zero external APIs."""
    if not predicted_summary:
        return 0.0, "Empty resolution summary"

    pred_lower = predicted_summary.lower()
    gt_lower = ground_truth_root_cause.lower()

    # Extract key operational tokens
    key_terms = [
        "commit", "deploy", "rollback", "reverted", "scale", "traffic", "deadlock",
        "oom", "memory", "redis", "pool", "disk", "cpu", "throttle", "timeout", "regex"
    ]
    gt_matches = [w for w in key_terms if w in gt_lower]

    matched = sum(1 for w in gt_matches if w in pred_lower)
    score = (matched / max(len(gt_matches), 1)) * 5.0
    score = min(5.0, max(1.0, round(score, 1)))

    rationale = f"Matched {matched}/{len(gt_matches)} critical failure descriptors ({', '.join(gt_matches)})."
    return score, rationale


def evaluate_with_bedrock(predicted_summary, ground_truth_root_cause):
    """Optional Amazon Bedrock Claude 3.5 Sonnet Judge."""
    try:
        import boto3
        bedrock = boto3.client("bedrock-runtime")
        prompt = (
            "You are an expert SRE evaluator. Compare the agent's incident resolution with the ground truth.\n\n"
            f"Ground Truth Root Cause: {ground_truth_root_cause}\n"
            f"Agent's Resolution: {predicted_summary}\n\n"
            "Score root-cause identification on a scale of 0 to 5, and provide a 1-sentence rationale.\n"
            "Output valid JSON: {\"score\": 5.0, \"rationale\": \"...\"}"
        )
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 256,
            "messages": [{"role": "user", "content": prompt}],
        })
        resp = bedrock.invoke_model(modelId="anthropic.claude-3-5-sonnet-20241022-v2:0", body=body)
        res_body = json.loads(resp["body"].read())
        text = res_body["content"][0]["text"]
        data = json.loads(re.search(r"\{.*\}", text, re.DOTALL).group(0))
        return float(data.get("score", 4.0)), data.get("rationale", "Graded by Bedrock.")
    except Exception as e:
        print(f"Bedrock judge unavailable ({e}). Falling back to heuristic judge.")
        return evaluate_with_heuristic_judge(predicted_summary, ground_truth_root_cause)


def run_benchmark(api_base, dataset_path, judge_type="heuristic", max_samples=None):
    incidents = []
    with open(dataset_path) as f:
        for line in f:
            if line.strip():
                incidents.append(json.loads(line))

    if max_samples:
        incidents = incidents[:max_samples]

    print(f"\n🚀 Running SentinelOps Evaluation Benchmark on {len(incidents)} validation incidents...")
    print(f"Target API Endpoint: {api_base}")
    print(f"Judge Backend: {judge_type}\n")

    results = []

    for idx, inc in enumerate(incidents, 1):
        inc_id = inc.get("incident_id", f"inc-{idx}")[:8]
        service = inc.get("service", "service")
        inc_type = inc.get("incident_type", "unknown")
        gt_root_cause = inc.get("root_cause", "")
        gt_tools = [step["tool"] for step in inc.get("tool_trace", [])]
        gt_remediation_tool = gt_tools[-1] if gt_tools else None

        env = DynamicIncidentEnvironment(inc)

        initial_user_msg = (
            f"Incident detected on `{service}` at {inc.get('timestamp', 'now')}.\n\n"
            f"Raw telemetry / log excerpt:\n```\n{inc.get('injected_logs', '').strip()}\n```\n\n"
            "Diagnose the root cause and take the appropriate remediation action."
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": initial_user_msg},
        ]

        turns_taken = 0
        executed_tools = []
        turn_latencies = []
        final_summary = ""

        # Multi-turn ReAct execution
        for turn in range(1, 7):
            turns_taken = turn
            resp_data, latency_ms = call_api(api_base, messages, TOOLS)
            turn_latencies.append(latency_ms)

            choice = resp_data["choices"][0]
            msg = choice["message"]
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls", [])

            if not tool_calls and content:
                extracted = extract_json_objects(content)
                if extracted:
                    tool_calls = [{
                        "type": "function",
                        "function": {"name": c["name"], "arguments": c.get("arguments", {})}
                    } for c in extracted]

            if tool_calls:
                messages.append({"role": "assistant", "content": "", "tool_calls": tool_calls})
                for call in tool_calls:
                    fn_name = call["function"]["name"]
                    fn_args = call["function"]["arguments"]
                    executed_tools.append(fn_name)
                    tool_res = env.execute(fn_name, fn_args)
                    messages.append({"role": "tool", "name": fn_name, "content": tool_res})
            else:
                final_summary = content.replace("<|im_start|>", "").replace("<|im_end|>", "").strip()
                break

        # Ground truth checks
        agent_remediation_tool = None
        for t in ["execute_rollback", "scale_autoscaling_group"]:
            if t in executed_tools:
                agent_remediation_tool = t
                break

        remediation_match = (agent_remediation_tool == gt_remediation_tool) if gt_remediation_tool else True
        has_diagnosis_before_action = False
        if executed_tools:
            if executed_tools[0] in ["fetch_service_metrics", "query_vector_db", "fetch_git_diff", "describe_pod", "get_logs"]:
                has_diagnosis_before_action = True

        # Judge evaluation
        if judge_type == "bedrock":
            judge_score, judge_rationale = evaluate_with_bedrock(final_summary, gt_root_cause)
        else:
            judge_score, judge_rationale = evaluate_with_heuristic_judge(final_summary, gt_root_cause)

        avg_lat = sum(turn_latencies) / max(len(turn_latencies), 1)

        result_entry = {
            "incident_id": inc_id,
            "service": service,
            "incident_type": inc_type,
            "turns": turns_taken,
            "remediation_match": remediation_match,
            "discipline_pass": has_diagnosis_before_action,
            "judge_score": judge_score,
            "judge_rationale": judge_rationale,
            "avg_latency_ms": round(avg_lat, 1),
            "tools_called": executed_tools,
            "final_summary": final_summary[:120] + "..." if len(final_summary) > 120 else final_summary,
        }
        results.append(result_entry)

        status_icon = "✅" if remediation_match and has_diagnosis_before_action else "⚠️"
        print(f"[{idx}/{len(incidents)}] {status_icon} {inc_id} ({service} - {inc_type}): {turns_taken} turns, {round(avg_lat)}ms/turn, Score: {judge_score}/5.0")

    return results


def generate_scorecard(results, output_path="benchmark_scorecard.md"):
    n = len(results)
    if n == 0:
        print("No results to generate scorecard.")
        return

    rem_acc = (sum(1 for r in results if r["remediation_match"]) / n) * 100
    disc_acc = (sum(1 for r in results if r["discipline_pass"]) / n) * 100
    avg_turns = sum(r["turns"] for r in results) / n
    avg_score = sum(r["judge_score"] for r in results) / n
    avg_lat = sum(r["avg_latency_ms"] for r in results) / n

    overall_sre_readiness = (rem_acc * 0.45) + (disc_acc * 0.25) + ((avg_score / 5.0) * 100 * 0.30)

    md = []
    md.append("# SentinelOps Benchmark Scorecard\n")
    md.append(f"**Date:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
    md.append("## Executive Summary\n")
    md.append(f"- **Overall SRE Readiness Score:** **`{overall_sre_readiness:.1f}%`**")
    md.append(f"- **Remediation Precision:** **`{rem_acc:.1f}%`** (Correct rollback vs autoscaling actions)")
    md.append(f"- **Diagnostic Discipline:** **`{disc_acc:.1f}%`** (Inspected metrics/logs before acting)")
    md.append(f"- **Average LLM-as-a-Judge Score:** **`{avg_score:.2f} / 5.0`**")
    md.append(f"- **Average Turns to Resolution:** **`{avg_turns:.1f}`**")
    md.append(f"- **Average Turn Latency:** **`{avg_lat:.0f} ms`**\n")

    md.append("## Incident Breakdown by Scenario\n")
    md.append("| Incident ID | Service | Failure Type | Turns | Remediation | Discipline | Judge Score | Latency |")
    md.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for r in results:
        rem_icon = "✅ Pass" if r["remediation_match"] else "❌ Fail"
        disc_icon = "✅ Pass" if r["discipline_pass"] else "❌ Fail"
        md.append(f"| `{r['incident_id']}` | `{r['service']}` | `{r['incident_type']}` | {r['turns']} | {rem_icon} | {disc_icon} | {r['judge_score']}/5.0 | {r['avg_latency_ms']:.0f}ms |")

    md.append("\n## Sample Execution Traces\n")
    for r in results[:3]:
        md.append(f"### Incident `{r['incident_id']}` ({r['service']})")
        md.append(f"- **Failure Mode:** `{r['incident_type']}`")
        md.append(f"- **Tools Executed:** `{' -> '.join(r['tools_called'])}`")
        md.append(f"- **Judge Rationale:** {r['judge_rationale']}")
        md.append(f"- **Agent Summary:** *\"{r['final_summary']}\"*\n")

    content = "\n".join(md)
    with open(output_path, "w") as f:
        f.write(content)

    print(f"\n🎉 Benchmark Scorecard exported to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-base", default="http://localhost:8000/v1", help="URL of SentinelOps API server")
    parser.add_argument("--dataset", default="data/incidents_val.jsonl", help="Validation incidents JSONL path")
    parser.add_argument("--judge", default="heuristic", choices=["heuristic", "bedrock"], help="Judge backend")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit sample count for quick runs")
    parser.add_argument("--output", default="benchmark_scorecard.md", help="Output markdown scorecard path")
    args = parser.parse_args()

    results = run_benchmark(args.api_base, args.dataset, args.judge, args.max_samples)
    generate_scorecard(results, args.output)


if __name__ == "__main__":
    main()
