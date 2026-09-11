"""
agent_orchestrator.py

Autonomous ReAct Agent Orchestrator for SentinelOps.
Connects to the SentinelOps model server (e.g., http://localhost:8000/v1), receives
incident alerts, executes investigative and remediation tools in a multi-turn
loop, and returns the final resolution report.

Usage:
    python agent_orchestrator.py --api-base http://localhost:8000/v1
    python agent_orchestrator.py --api-base http://localhost:8000/v1 --scenario oom_kill
"""
import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
try:
    from common.tool_schemas import SYSTEM_PROMPT, TOOLS
except ImportError:
    from tool_schemas import SYSTEM_PROMPT, TOOLS


# Built-in incident scenarios for interactive testing
TEST_SCENARIOS = {
    "db_deadlock": {
        "service": "checkout-service",
        "alert": (
            "Incident detected on `checkout-service` at 2026-09-11T06:00:00Z.\n\n"
            "Raw telemetry / log excerpt:\n```\n"
            "ERROR: postgres: deadlock detected on tables `inventory` and `orders`.\n"
            "Process 4189 waits for ShareLock on transaction 83468; blocked by process 4437.\n"
            "HTTP 500 InternalServerError for POST /orders/checkout.\n```\n\n"
            "Diagnose the root cause and take the appropriate remediation action."
        ),
        "metrics": {"db_deadlock_count": "184 deadlock occurrences in the last 30 minutes; spike started after deploy at 05:45Z."},
        "git_diff": "diff --git a/orders.py b/orders.py\n- db.lock('inventory')\n- db.lock('orders')\n+ db.lock('orders')\n+ db.lock('inventory')",
        "runbook": "Runbook DB-007: Circular locking between orders and inventory tables indicates a lock-ordering code regression. Execute rollback immediately.",
    },
    "oom_kill": {
        "service": "auth-service",
        "alert": (
            "Incident detected on `auth-service` at 2026-09-11T06:15:00Z.\n\n"
            "Raw telemetry / log excerpt:\n```\n"
            "WARN: Redis cluster memory usage 98.4%. Keyspace eviction rate 45,000 keys/sec.\n"
            "HTTP 429 & 503 on /oauth/token verification.\n"
            "Pod auth-service-7df9f9-x2k9l restarts: 4 within 10 minutes.\n```\n\n"
            "Diagnose the root cause and take the appropriate remediation action."
        ),
        "pod_status": "Pod: auth-service-7df9f9-x2k9l | State: CrashLoopBackOff | Last State: Terminated with exit code 137 (OOMKilled) | Memory Limit: 2.0Gi (Hit 100%).",
        "metrics": {"memory_utilization": "memory_utilization=0.99 across all auth-service pods; Redis connection pool usage=100%."},
        "runbook": "Runbook AUTH-012: In case of cache eviction storms, scale out the autoscaling group to distribute heap load while cache warms.",
    },
    "traffic_surge": {
        "service": "notification-worker",
        "alert": (
            "Incident detected on `notification-worker` at 2026-09-11T06:30:00Z.\n\n"
            "Raw telemetry / log excerpt:\n```\n"
            "ALERT: Kafka Consumer Group 'notification-dispatch' lag breached 600,000 messages.\n"
            "Processing latency p95: 160s (SLO < 5s).\n"
            "Zero application crash errors reported.\n```\n\n"
            "Diagnose the root cause and take the appropriate remediation action."
        ),
        "metrics": {"request_rate": "Incoming message rate jumped 8x from 1,200 msg/s to 9,800 msg/s starting at 06:15Z."},
        "git_diff": "No recent commits deployed in the last 48 hours. Service running stable build v1.8.0.",
        "runbook": "Runbook KAFKA-03: If incoming message rate spikes without code changes, scale autoscaling group to drain lag.",
    }
}


class ToolDispatcher:
    """Executes SRE diagnostic and remediation tools."""

    def __init__(self, scenario_data=None):
        self.scenario = scenario_data or {}
        self.runbooks = self._load_runbooks()

    def _load_runbooks(self):
        runbooks = []
        rb_path = Path("data/runbooks.jsonl")
        if rb_path.exists():
            with open(rb_path) as f:
                for line in f:
                    if line.strip():
                        runbooks.append(json.loads(line))
        return runbooks

    def execute(self, tool_name, arguments):
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except Exception:
                pass

        if not isinstance(arguments, dict):
            arguments = {}

        print(f"\n⚙️  [Tool Execution] -> {tool_name}({json.dumps(arguments)})")

        if tool_name == "fetch_service_metrics":
            metric_name = arguments.get("metric_name", "")
            time_range = arguments.get("time_range", "30m")
            if "metrics" in self.scenario and metric_name in self.scenario["metrics"]:
                res = self.scenario["metrics"][metric_name]
            else:
                res = f"Metric '{metric_name}' over {time_range}: elevated variance observed (current value 94.2% threshold breach)."
            print(f"📊 [Result]: {res}")
            return res

        elif tool_name == "query_vector_db":
            query = arguments.get("query", "").lower()
            if "runbook" in self.scenario:
                res = self.scenario["runbook"]
            elif self.runbooks:
                # Find matching runbook by keyword
                matches = [rb["remediation_procedure"] for rb in self.runbooks if any(w in query for w in rb.get("tags", []))]
                res = matches[0] if matches else self.runbooks[0]["content"]
            else:
                res = f"Runbook matching '{query}': Check git diff for recent deploy regressions or scale ASG if traffic is organic."
            print(f"📖 [Result]: {res}")
            return res

        elif tool_name == "fetch_git_diff":
            service = arguments.get("service", "target-service")
            commit = arguments.get("commit_hash", "HEAD")
            if "git_diff" in self.scenario:
                res = self.scenario["git_diff"]
            else:
                res = f"Commit {commit} on {service}: PR #849 updated lock ordering in transaction handler."
            print(f"🔍 [Result]:\n{res}")
            return res

        elif tool_name == "describe_pod":
            pod = arguments.get("pod_name", "service-pod")
            res = self.scenario.get("pod_status", f"Pod {pod}: Running (1 restart in last 10m).")
            print(f"📦 [Result]: {res}")
            return res

        elif tool_name == "get_logs":
            service = arguments.get("service", "service")
            res = f"Log stream for {service}: Multiple transaction rollback errors; worker thread timeout."
            print(f"📜 [Result]: {res}")
            return res

        elif tool_name == "execute_rollback":
            service_id = arguments.get("service_id", "target-service")
            res = f"Rollback initiated successfully for `{service_id}`. Reverted to previous stable revision. Traffic healthy."
            print(f"✅ [Result]: {res}")
            return res

        elif tool_name == "scale_autoscaling_group":
            asg = arguments.get("asg_name", "asg")
            cap = arguments.get("target_capacity", 8)
            res = f"Autoscaling group `{asg}` successfully scaled to {cap} replicas. Provisioning complete."
            print(f"🚀 [Result]: {res}")
            return res

        else:
            res = f"Tool '{tool_name}' executed successfully."
            print(f"ℹ️  [Result]: {res}")
            return res


def call_model_api(api_base, messages, tools):
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

    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def run_agent_loop(api_base, scenario_name="db_deadlock", max_turns=6):
    scenario = TEST_SCENARIOS.get(scenario_name, TEST_SCENARIOS["db_deadlock"])
    dispatcher = ToolDispatcher(scenario)

    print("\n" + "=" * 70)
    print(f"🚨 [SENTINELOPS RE-ACT AGENT TRIGGERED: Scenario '{scenario_name}']")
    print("=" * 70)
    print(scenario["alert"])
    print("=" * 70)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": scenario["alert"]},
    ]

    for turn in range(1, max_turns + 1):
        print(f"\n--- [Turn {turn}/{max_turns}] Waiting for SentinelOps agent decision ---")
        response = call_model_api(api_base, messages, TOOLS)
        choice = response["choices"][0]
        message = choice["message"]

        content = message.get("content", "")
        tool_calls = message.get("tool_calls", [])

        # Check if the model emitted a tool call
        if tool_calls:
            messages.append({
                "role": "assistant",
                "content": content or "",
                "tool_calls": tool_calls,
            })

            for call in tool_calls:
                fn_name = call["function"]["name"]
                fn_args = call["function"]["arguments"]
                tool_result = dispatcher.execute(fn_name, fn_args)

                messages.append({
                    "role": "tool",
                    "name": fn_name,
                    "content": tool_result,
                })

        # Check if model produced raw JSON call in content
        elif content and "{" in content and "name" in content and "arguments" in content:
            try:
                call_json = json.loads(content.replace("<|im_start|>", "").replace("<|im_end|>", "").strip())
                fn_name = call_json["name"]
                fn_args = call_json["arguments"]
                tool_result = dispatcher.execute(fn_name, fn_args)

                messages.append({
                    "role": "assistant",
                    "content": content,
                })
                messages.append({
                    "role": "tool",
                    "name": fn_name,
                    "content": tool_result,
                })
            except Exception:
                # Text resolution
                print("\n" + "=" * 70)
                print("🎯 [INCIDENT RESOLUTION REPORT]")
                print("=" * 70)
                print(content.replace("<|im_start|>", "").replace("<|im_end|>", "").strip())
                print("=" * 70)
                break
        else:
            # Final text resolution reached
            print("\n" + "=" * 70)
            print("🎯 [INCIDENT RESOLUTION REPORT]")
            print("=" * 70)
            print(content.replace("<|im_start|>", "").replace("<|im_end|>", "").strip())
            print("=" * 70)
            break

        time.sleep(0.5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-base", default="http://localhost:8000/v1", help="URL of SentinelOps API server")
    parser.add_argument("--scenario", default="db_deadlock", choices=list(TEST_SCENARIOS.keys()), help="Scenario to simulate")
    parser.add_argument("--max-turns", type=int, default=6, help="Maximum investigation turns")
    args = parser.parse_args()

    run_agent_loop(args.api_base, args.scenario, args.max_turns)


if __name__ == "__main__":
    main()
