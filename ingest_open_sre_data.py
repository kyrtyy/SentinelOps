"""
ingest_open_sre_data.py

Ingests open-source SRE incident trajectories (such as Hugging Face's
`quantranger/opensre-incident-trajectories` and curated real-world outage
investigations from AWS, Cloudflare, Slack, GitHub, Datadog), normalizes
them into SentinelOps's multi-turn tool-calling schema, and blends them
with synthetic training data.

Output format for each line in sft_train.jsonl / sft_val.jsonl:
    {"messages": [...], "tools": [...]}
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
try:
    from common.tool_schemas import SYSTEM_PROMPT, TOOLS
except ImportError:
    from tool_schemas import SYSTEM_PROMPT, TOOLS

# Curated real-world incident benchmark trajectories inspired by public post-mortems
REAL_WORLD_BENCHMARK_INCIDENTS = [
    {
        "service": "api-gateway",
        "company_scenario": "Cloudflare / Edge Proxy Regex CPU Spike",
        "timestamp": "2026-06-18T13:42:10Z",
        "injected_logs": (
            "[CRITICAL] p99 latency spiked from 12ms to 4800ms across edge clusters.\n"
            "WAF rule engine worker pool CPU 100% pegged.\n"
            "Upstream connection timeouts: HTTP 524 / 504 on /api/v2/* endpoints."
        ),
        "tool_trace": [
            {
                "tool": "fetch_service_metrics",
                "arguments": {"metric_name": "cpu_throttle_ratio", "time_range": "15m"},
                "result": "cpu_throttle_ratio=0.98; core utilization 100% across all 64 worker threads."
            },
            {
                "tool": "get_logs",
                "arguments": {"service": "api-gateway", "filter_pattern": "WAF", "limit": 10},
                "result": "WAF evaluation timeout executing rule_id=94101 (regex backtracking detected on user-agent header)."
            },
            {
                "tool": "fetch_git_diff",
                "arguments": {"service": "api-gateway", "commit_hash": "HEAD"},
                "result": (
                    "diff --git a/waf/rules.yaml b/waf/rules.yaml\n"
                    "+ - id: 94101\n"
                    "+   pattern: '^(.*(?:[a-zA-Z0-9_-]+)+)+$'\n"
                    "+   action: BLOCK"
                )
            },
            {
                "tool": "execute_rollback",
                "arguments": {"service_id": "api-gateway"},
                "result": "Rollback completed to revision v4.19.2. WAF ruleset reloaded."
            }
        ],
        "resolution_summary": (
            "Root Cause: Catastrophic regular expression backtracking in newly deployed WAF rule 94101 "
            "causing 100% CPU starvation across worker threads.\n"
            "Remediation: Rolled back api-gateway to revision v4.19.2. CPU and latency returned to baseline."
        )
    },
    {
        "service": "checkout-service",
        "company_scenario": "E-Commerce Connection Pool Deadlock & Saturation",
        "timestamp": "2026-07-02T19:15:00Z",
        "injected_logs": (
            "ERROR: DB connection timeout after 5000ms: pool exhausted (max=200, in_use=200, queue_depth=842).\n"
            "HTTP 500 InternalServerError for POST /orders/checkout.\n"
            "CircuitBreaker 'postgres-primary' tripped OPEN."
        ),
        "tool_trace": [
            {
                "tool": "fetch_service_metrics",
                "arguments": {"metric_name": "db_deadlock_count", "time_range": "30m"},
                "result": "db_deadlock_count=184 spikes starting at 19:05Z; connection queue depth > 800."
            },
            {
                "tool": "query_vector_db",
                "arguments": {"query": "checkout-service db connection pool exhaustion deadlock runbook"},
                "result": (
                    "Runbook RB-402: Under high unindexed query volume or missing transaction timeouts, "
                    "transactions hold connections open. Check recent schema/query changes in git diff or scale ASG."
                )
            },
            {
                "tool": "fetch_git_diff",
                "arguments": {"service": "checkout-service", "commit_hash": "HEAD"},
                "result": (
                    "diff --git a/repo/orders.go b/repo/orders.go\n"
                    "- db.SetConnMaxLifetime(10 * time.Minute)\n"
                    "+ // Removed timeout to allow long analytics sync"
                )
            },
            {
                "tool": "execute_rollback",
                "arguments": {"service_id": "checkout-service"},
                "result": "Rollback successful for checkout-service. Reverted to v2.10.1."
            }
        ],
        "resolution_summary": (
            "Root Cause: Commit HEAD removed DB connection max lifetime timeout, causing unclosed transactions "
            "to permanently hold database connections and deadlock the pool.\n"
            "Remediation: Executed rollback to stable revision v2.10.1."
        )
    },
    {
        "service": "auth-service",
        "company_scenario": "Token Verification Cache Eviction Surge",
        "timestamp": "2026-08-11T09:20:45Z",
        "injected_logs": (
            "WARN: Redis cluster memory usage 98.4%. Keyspace eviction rate 45,000 keys/sec.\n"
            "HTTP 429 & 503 on /oauth/token verification.\n"
            "Pod auth-service-7df9f9-x2k9l restarts: 4 within 10 minutes."
        ),
        "tool_trace": [
            {
                "tool": "describe_pod",
                "arguments": {"pod_name": "auth-service-7df9f9-x2k9l", "namespace": "production"},
                "result": "Pod Status: CrashLoopBackOff. Last state: Terminated with Exit Code 137 (OOMKilled)."
            },
            {
                "tool": "fetch_service_metrics",
                "arguments": {"metric_name": "memory_utilization", "time_range": "30m"},
                "result": "memory_utilization=0.99 (hit 2.0Gi cgroup limit); Redis connection pool usage=100%."
            },
            {
                "tool": "scale_autoscaling_group",
                "arguments": {"asg_name": "auth-service-asg", "target_capacity": 12},
                "result": "auth-service-asg scaling initiated: target capacity increased from 4 to 12 instances."
            }
        ],
        "resolution_summary": (
            "Root Cause: Redis cache eviction storm caused auth pods to suffer cache stampedes, exhausting local heap "
            "and triggering OOMKills.\n"
            "Remediation: Scaled auth-service-asg capacity from 4 to 12 replicas to distribute memory load while cache repopulates."
        )
    },
    {
        "service": "search-indexer",
        "company_scenario": "Elasticsearch Disk Spill & Shard Relocation Storm",
        "timestamp": "2026-08-25T16:10:00Z",
        "injected_logs": (
            "Cluster health: RED (24 unassigned shards).\n"
            "Data node data-03 disk watermark 92% exceeded (high_disk_watermark=90%).\n"
            "Indexing rate dropped to 0 docs/sec (read-only index block enabled)."
        ),
        "tool_trace": [
            {
                "tool": "fetch_service_metrics",
                "arguments": {"metric_name": "disk_utilization", "time_range": "1h"},
                "result": "disk_utilization=0.93 on data-03; total cluster storage 89% full."
            },
            {
                "tool": "query_vector_db",
                "arguments": {"query": "Elasticsearch disk flood stage read-only index remediation"},
                "result": (
                    "Runbook ES-12: When disk watermark >90%, cluster locks indices to read-only. "
                    "Scale the storage volume or scale out cluster nodes immediately to unblock shards."
                )
            },
            {
                "tool": "scale_autoscaling_group",
                "arguments": {"asg_name": "es-data-nodes-asg", "target_capacity": 8},
                "result": "es-data-nodes-asg scaled from 5 to 8 nodes. EBS volumes provisioning."
            }
        ],
        "resolution_summary": (
            "Root Cause: Disk usage on node data-03 breached high watermark threshold, forcing indices into read-only mode.\n"
            "Remediation: Scaled es-data-nodes-asg to 8 instances to trigger shard rebalancing and reduce per-node storage pressure."
        )
    },
    {
        "service": "notification-worker",
        "company_scenario": "Kafka Consumer Lag Surge after Marketing Blast",
        "timestamp": "2026-09-01T11:05:30Z",
        "injected_logs": (
            "ALERT: Kafka Consumer Group 'notification-dispatch' lag breached 500,000 messages.\n"
            "Processing latency p95: 142s (SLO < 5s).\n"
            "Zero application crash errors reported."
        ),
        "tool_trace": [
            {
                "tool": "fetch_service_metrics",
                "arguments": {"metric_name": "request_rate", "time_range": "1h"},
                "result": "Incoming message rate jumped 8x from 1,200 msg/s to 9,800 msg/s starting at 11:00Z."
            },
            {
                "tool": "fetch_git_diff",
                "arguments": {"service": "notification-worker", "commit_hash": "HEAD"},
                "result": "No deployments in the last 24 hours. Service running stable build v1.8.0."
            },
            {
                "tool": "scale_autoscaling_group",
                "arguments": {"asg_name": "notification-worker-asg", "target_capacity": 20},
                "result": "Scaled notification-worker-asg from 4 to 20 workers."
            }
        ],
        "resolution_summary": (
            "Root Cause: Organic traffic surge (8x increase in message volume) without any code regression.\n"
            "Remediation: Scaled out consumer group workers from 4 to 20 to drain Kafka lag."
        )
    }
]


def format_incident_to_messages(incident):
    """Formats an incident dict into multi-turn messages array."""
    user_content = (
        f"Incident detected on `{incident['service']}` at {incident['timestamp']}.\n\n"
        f"Raw telemetry / log excerpt:\n```\n{incident['injected_logs'].strip()}\n```\n\n"
        "Diagnose the root cause and take the appropriate remediation action."
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    for step in incident["tool_trace"]:
        args = step["arguments"]
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                pass
        messages.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "type": "function",
                "function": {
                    "name": step["tool"],
                    "arguments": args,
                },
            }],
        })
        messages.append({
            "role": "tool",
            "name": step["tool"],
            "content": step["result"],
        })

    messages.append({
        "role": "assistant",
        "content": incident["resolution_summary"],
    })
    return messages


def fetch_hf_opensre_dataset(dataset_name="quantranger/opensre-incident-trajectories"):
    """Attempts to fetch OpenSRE dataset from Hugging Face if available."""
    try:
        from datasets import load_dataset
        print(f"Attempting to fetch '{dataset_name}' from Hugging Face Hub...")
        ds = load_dataset(dataset_name, split="train")
        incidents = []
        for row in ds:
            if "messages" in row:
                incidents.append({"messages": row["messages"], "tools": TOOLS})
            elif "trajectory" in row or "tool_trace" in row:
                incidents.append({
                    "service": row.get("service", "production-service"),
                    "timestamp": row.get("timestamp", "2026-09-10T00:00:00Z"),
                    "injected_logs": row.get("injected_logs", row.get("alert", "Alert triggered")),
                    "tool_trace": row.get("tool_trace", row.get("trajectory", [])),
                    "resolution_summary": row.get("resolution_summary", row.get("root_cause", "Resolved.")),
                })
        print(f"Successfully loaded {len(incidents)} examples from Hugging Face.")
        return incidents
    except Exception as e:
        print(f"Note: Could not fetch from Hugging Face ({e}). Using curated open SRE benchmark suite.")
        return []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data", help="Output directory for JSONL datasets")
    parser.add_argument("--synthetic-train", default="data/incidents_train.jsonl", help="Path to synthetic train data")
    parser.add_argument("--synthetic-val", default="data/incidents_val.jsonl", help="Path to synthetic val data")
    parser.add_argument("--include-hf", action="store_true", help="Attempt to pull quantranger/opensre-incident-trajectories")
    parser.add_argument("--val-split", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for data shuffling")
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_records = []

    # 1. Load curated real-world benchmark incidents
    for inc in REAL_WORLD_BENCHMARK_INCIDENTS:
        all_records.append({
            "messages": format_incident_to_messages(inc),
            "tools": TOOLS,
        })

    # 2. Optionally load Hugging Face dataset
    if args.include_hf:
        hf_data = fetch_hf_opensre_dataset()
        for item in hf_data:
            if "messages" in item:
                all_records.append(item)
            else:
                all_records.append({
                    "messages": format_incident_to_messages(item),
                    "tools": TOOLS,
                })

    # 3. Load synthetic incidents if they exist
    synth_train_path = Path(args.synthetic_train)
    if synth_train_path.exists():
        with open(synth_train_path) as f:
            for line in f:
                if line.strip():
                    inc = json.loads(line)
                    all_records.append({
                        "messages": format_incident_to_messages(inc),
                        "tools": TOOLS,
                    })
        print(f"Loaded synthetic records from {synth_train_path}")

    random.shuffle(all_records)
    n_total = len(all_records)
    n_val = max(1, int(n_total * args.val_split))
    n_train = n_total - n_val

    train_records = all_records[:n_train]
    val_records = all_records[n_train:]

    train_path = output_dir / "sft_train.jsonl"
    val_path = output_dir / "sft_val.jsonl"

    with open(train_path, "w") as f:
        for r in train_records:
            f.write(json.dumps(r) + "\n")

    with open(val_path, "w") as f:
        for r in val_records:
            f.write(json.dumps(r) + "\n")

    print(f"Wrote {len(train_records)} training records to {train_path}")
    print(f"Wrote {len(val_records)} validation records to {val_path}")


if __name__ == "__main__":
    main()
