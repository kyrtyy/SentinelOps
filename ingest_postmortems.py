"""
ingest_postmortems.py

Curates and chunks public SRE post-mortems and operational runbooks for
SentinelOps's pgvector RAG knowledge base. Ingests raw post-mortem markdown
or structured records and outputs normalized chunks with semantic search metadata
for `query_vector_db`.
"""
import argparse
import json
from pathlib import Path

CURATED_POSTMORTEMS = [
    {
        "id": "PM-001",
        "title": "Cloudflare Global WAF CPU Exhaustion Outage",
        "service": "api-gateway",
        "tags": ["waf", "cpu_spike", "regex", "rollback"],
        "root_cause": "Unchecked catastrophic backtracking in newly deployed regular expression inside WAF rule engine.",
        "remediation_procedure": "1. Immediately identify newly deployed WAF rules via fetch_git_diff. 2. Execute rollback to previous revision. 3. Reload rule engine.",
        "content": (
            "# Cloudflare Global WAF Outage Post-Mortem\n"
            "Summary: A single regular expression rule deployed to edge WAF caused 100% CPU usage across all worker threads.\n"
            "Detection: p99 latency jumped 400x; HTTP 524 timeouts.\n"
            "Fix: Immediate rollback of edge rule repository restored traffic within 3 minutes."
        )
    },
    {
        "id": "PM-002",
        "title": "Database Connection Pool Exhaustion & Deadlock",
        "service": "checkout-service",
        "tags": ["database", "postgres", "connection_pool", "deadlock"],
        "root_cause": "Long-lived uncommitted transactions holding connection slots combined with removed connection lifetime timeouts.",
        "remediation_procedure": "1. Check db_deadlock_count and active connection ratio. 2. Inspect recent connection pool configuration changes. 3. Revert configuration or restart service.",
        "content": (
            "# DB Connection Pool Exhaustion Post-Mortem\n"
            "Summary: Microservice connection pool exhausted during peak sales event.\n"
            "Root Cause: Database driver timeout removed in PR #1142.\n"
            "Resolution: Rolled back checkout service deployment."
        )
    },
    {
        "id": "PM-003",
        "title": "Kubernetes OOMKilled Cascading Restart Storm",
        "service": "auth-service",
        "tags": ["kubernetes", "oomkill", "memory", "autoscaling"],
        "root_cause": "Cache miss storm increased heap usage beyond pod cgroup memory limits.",
        "remediation_procedure": "1. Describe affected pods to confirm OOMKilled exit code 137. 2. Scale up autoscaling group/replica count to distribute load. 3. Increase pod memory limits in deployment manifest.",
        "content": (
            "# Auth Service OOMKill Cascading Storm\n"
            "Summary: Redis cache eviction caused auth pods to flood backend and run out of memory.\n"
            "Fix: Scaled autoscaling group replicas from 4 to 12 to relieve memory pressure."
        )
    },
    {
        "id": "PM-004",
        "title": "Kafka Consumer Group Lag Breaching SLO",
        "service": "notification-worker",
        "tags": ["kafka", "queue_lag", "traffic_surge", "scaling"],
        "root_cause": "8x traffic surge from scheduled marketing campaign overwhelmed existing 4 worker pods.",
        "remediation_procedure": "1. Verify whether code changes occurred with fetch_git_diff. 2. Check incoming message rate. 3. If organic traffic, scale autoscaling group to match throughput.",
        "content": (
            "# Notification Worker Lag Post-Mortem\n"
            "Summary: Massive consumer lag spike with zero application errors.\n"
            "Fix: Scaled autoscaling group from 4 to 20 workers."
        )
    },
    {
        "id": "PM-005",
        "title": "Elasticsearch High Disk Watermark Read-Only Lock",
        "service": "search-indexer",
        "tags": ["elasticsearch", "disk_full", "read_only", "storage"],
        "root_cause": "Daily index rollover disk usage exceeded 90% high watermark.",
        "remediation_procedure": "1. Query disk_utilization metrics. 2. Add storage capacity or scale cluster data nodes.",
        "content": (
            "# Elasticsearch Disk Flood Stage Outage\n"
            "Summary: Cluster blocked writes due to disk threshold breach.\n"
            "Fix: Scaled data nodes ASG from 5 to 8 nodes to trigger shard rebalancing."
        )
    }
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/runbooks.jsonl", help="Output path for runbook JSONL")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        for doc in CURATED_POSTMORTEMS:
            f.write(json.dumps(doc) + "\n")

    print(f"Exported {len(CURATED_POSTMORTEMS)} post-mortems and runbooks to {out_path}")


if __name__ == "__main__":
    main()
