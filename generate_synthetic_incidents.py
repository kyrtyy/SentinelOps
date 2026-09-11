"""
generate_synthetic_incidents.py

Generates a synthetic corpus of SRE incident scenarios for SentinelOps.
Each scenario pairs a raw telemetry/log excerpt with a ground-truth
root cause and the ordered tool-call trace a correct triage agent
should produce.

Deliberately template + randomization based (no external LLM/API
calls): the dataset is free, fully offline, and exactly reproducible
given a --seed, which matters for a repo other people will actually
try to run.

Output: JSONL, one raw incident record per line. Run
format_tool_calling_dataset.py next to turn these into trainable
multi-turn tool-calling conversations.

Usage:
    python generate_synthetic_incidents.py \
        --num-incidents 600 --output-dir data --seed 13
"""
import argparse
import json
import os
import random
import uuid
from datetime import datetime, timedelta, timezone

SERVICES = [
    "checkout-service", "payments-api", "auth-service", "inventory-service",
    "notification-worker", "search-indexer", "orders-service", "session-service",
    "recommendation-engine", "shipping-calculator",
]


def rand_commit_hash(rng):
    return "".join(rng.choice("0123456789abcdef") for _ in range(7))


def rand_timestamp(rng, start_year=2025, end_year=2026):
    start = datetime(start_year, 1, 1, tzinfo=timezone.utc)
    end = datetime(end_year, 12, 31, tzinfo=timezone.utc)
    delta_seconds = int((end - start).total_seconds())
    return start + timedelta(seconds=rng.randint(0, delta_seconds))


def fmt(ts):
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------
# Per-incident-type generators. Each returns a dict with injected_logs,
# root_cause, tool_trace (ordered list of {tool, arguments, result}), and
# resolution_summary. The tool_trace pattern is deliberately consistent
# across types (metrics -> runbook -> [git diff] -> remediation) so the
# model learns a repeatable investigation habit rather than memorizing
# per-incident-type shortcuts.
# --------------------------------------------------------------------------

def _gen_http_5xx_spike(rng, service, ts, commit):
    baseline = round(rng.uniform(0.1, 0.5), 2)
    spike = round(rng.uniform(12.0, 25.0), 1)
    n_lines = rng.randint(4, 7)
    lines = []
    for i in range(n_lines):
        t = ts + timedelta(seconds=i * 3)
        req_id = uuid.uuid4().hex[:12]
        lines.append(f"{fmt(t)} [ERROR] {service}: 502 Bad Gateway (upstream connect error) request_id={req_id}")
    lines.append(
        f"{fmt(ts + timedelta(seconds=(n_lines + 1) * 3))} [WARN] alertmanager: "
        f"5xx error rate for {service} exceeded threshold ({spike}% over 5m, baseline {baseline}%)"
    )
    logs = "\n".join(lines)

    root_cause = (
        f"Deploy {commit} reduced the upstream payment-gateway client timeout from 5000ms to 500ms, "
        f"causing legitimate slow responses to be treated as connection failures under normal load."
    )
    tool_trace = [
        {"tool": "fetch_service_metrics",
         "arguments": {"metric_name": "http_5xx_rate", "time_range": "15m"},
         "result": (f"5xx error rate for {service} spiked to {spike}% at {fmt(ts)}, up from a {baseline}% "
                    f"baseline. CPU and memory are nominal; no infra-side alerts fired.")},
        {"tool": "query_vector_db",
         "arguments": {"query": f"{service} 502 bad gateway spike runbook"},
         "result": ("Runbook 'SVC-014: Elevated 5xx Rate' recommends checking the most recent deploy diff "
                    "before assuming an infra cause, since CPU/memory look nominal here.")},
        {"tool": "fetch_git_diff",
         "arguments": {"service": service, "commit_hash": commit},
         "result": (f"Commit {commit}: changed `PAYMENT_GATEWAY_TIMEOUT_MS` from 5000 to 500 in "
                    f"payment_gateway_client.py. No other changes in this diff.")},
        {"tool": "execute_rollback",
         "arguments": {"service_id": service},
         "result": f"Rollback initiated. {service} now serving from the previous stable revision (pre-{commit})."},
    ]
    resolution = (
        f"Root cause: commit {commit} cut the payment-gateway client timeout to 500ms, well under normal "
        f"upstream latency, so healthy responses were being counted as failures. Rolled {service} back to "
        f"the prior revision; 5xx rate returned to the {baseline}% baseline within a few minutes."
    )
    return {"injected_logs": logs, "root_cause": root_cause, "tool_trace": tool_trace, "resolution_summary": resolution}


def _gen_oom_kill(rng, service, ts, commit):
    pid = rng.randint(1000, 9999)
    rss_mb = rng.randint(3800, 4090)
    restarts = rng.randint(3, 6)
    lines = [
        f"{fmt(ts)} [ERROR] kernel: Out of memory: Killed process {pid} ({service}-worker) "
        f"total-vm:{rss_mb + 300}212kB, anon-rss:{rss_mb}112kB",
        f"{fmt(ts + timedelta(seconds=1))} [ERROR] {service}: worker pod "
        f"{service}-{commit}-{rng.randint(1000, 9999)} OOMKilled (exit code 137)",
        f"{fmt(ts + timedelta(seconds=2))} [WARN] k8s: pod restart count for {service} now {restarts} in 10 minutes",
    ]
    logs = "\n".join(lines)
    root_cause = (
        f"Commit {commit} added an in-process cache in {service} with no eviction policy or TTL, "
        f"causing memory to grow until the container hit its resource limit."
    )
    tool_trace = [
        {"tool": "fetch_service_metrics",
         "arguments": {"metric_name": "memory_utilization", "time_range": "1h"},
         "result": (f"Memory usage for {service} climbs roughly linearly from 40% to 98% over the last "
                    f"~50 minutes, consistent with an unbounded cache rather than a request-volume spike "
                    f"(request rate is flat).")},
        {"tool": "query_vector_db",
         "arguments": {"query": f"OOMKilled {service} memory leak runbook"},
         "result": ("Runbook 'OOM-002: Diagnosing Memory Leaks' recommends checking recent deploys for new "
                    "in-memory caching or buffering logic before scaling up, since scaling only delays the "
                    "eventual OOM.")},
        {"tool": "fetch_git_diff",
         "arguments": {"service": service, "commit_hash": commit},
         "result": (f"Commit {commit}: introduced `_RESPONSE_CACHE = {{}}` (plain dict, no maxsize/TTL) in "
                    f"{service}/cache.py, populated on every request.")},
        {"tool": "execute_rollback",
         "arguments": {"service_id": service},
         "result": (f"Rollback initiated. {service} now serving from the previous stable revision "
                    f"(pre-{commit}); memory usage stabilizing.")},
    ]
    resolution = (
        f"Root cause: commit {commit} introduced an unbounded in-memory response cache with no eviction, "
        f"so RSS grew until the pod's memory limit triggered repeated OOM kills. Rolled {service} back; "
        f"memory usage has flattened out at its normal baseline."
    )
    return {"injected_logs": logs, "root_cause": root_cause, "tool_trace": tool_trace, "resolution_summary": resolution}


def _gen_db_deadlock(rng, service, ts, commit):
    tx1, tx2 = rng.randint(80000, 89999), rng.randint(80000, 89999)
    pid1, pid2 = rng.randint(3000, 4999), rng.randint(3000, 4999)
    lines = [
        f"{fmt(ts)} [ERROR] postgres: deadlock detected",
        f"DETAIL: Process {pid1} waits for ShareLock on transaction {tx1}; blocked by process {pid2}.",
        f"        Process {pid2} waits for ShareLock on transaction {tx2}; blocked by process {pid1}.",
        f"{fmt(ts)} [ERROR] {service}: transaction rolled back due to deadlock "
        f"(statement: UPDATE inventory SET qty = qty - 1 WHERE sku = ?)",
    ]
    logs = "\n".join(lines)
    root_cause = (
        f"Commit {commit} added a batch-update path in {service} that acquires row locks on `inventory` "
        f"then `orders` in the opposite order to the existing checkout path, producing circular waits "
        f"under concurrent load."
    )
    tool_trace = [
        {"tool": "fetch_service_metrics",
         "arguments": {"metric_name": "db_deadlock_count", "time_range": "30m"},
         "result": (f"Deadlock count for the database backing {service} rose from ~0/hour to 14 in the "
                    f"last 30 minutes, all involving the `inventory` and `orders` tables.")},
        {"tool": "query_vector_db",
         "arguments": {"query": "postgres deadlock inventory orders lock order runbook"},
         "result": ("Runbook 'DB-007: Deadlocks on inventory/orders' notes this pattern is almost always a "
                    "lock-ordering regression introduced by a new code path, not a load issue.")},
        {"tool": "fetch_git_diff",
         "arguments": {"service": service, "commit_hash": commit},
         "result": (f"Commit {commit}: new `bulk_adjust_inventory()` locks `orders` before `inventory`; "
                    f"the existing checkout path locks `inventory` before `orders`.")},
        {"tool": "execute_rollback",
         "arguments": {"service_id": service},
         "result": f"Rollback initiated. {service} now serving from the previous stable revision (pre-{commit})."},
    ]
    resolution = (
        f"Root cause: commit {commit} introduced a new bulk-update path that takes the `inventory`/`orders` "
        f"row locks in reverse order relative to the checkout path, creating circular waits. Rolled back "
        f"{service}; deadlock rate returned to baseline."
    )
    return {"injected_logs": logs, "root_cause": root_cause, "tool_trace": tool_trace, "resolution_summary": resolution}


def _gen_redis_timeout(rng, service, ts, commit):
    active = 200
    growth_x = round(rng.uniform(2.5, 4.0), 1)
    lines = [
        f"{fmt(ts)} [ERROR] {service}: redis.exceptions.TimeoutError: Timeout reading from socket",
        f"{fmt(ts)} [WARN] redis-pool: connection pool exhausted (active={active}, max={active})",
        f"{fmt(ts + timedelta(seconds=5))} [ERROR] {service}: 12 requests failed with redis timeout in the last 10s",
    ]
    logs = "\n".join(lines)
    root_cause = (
        f"Organic traffic to {service} grew {growth_x}x over the last week (no recent deploy), and the "
        f"shared Redis connection pool was never resized to match, so it saturates during peak hours."
    )
    tool_trace = [
        {"tool": "fetch_service_metrics",
         "arguments": {"metric_name": "redis_connection_pool_usage", "time_range": "1h"},
         "result": (f"Redis connection pool for {service} has been pinned at 100% utilization "
                    f"({active}/{active}) for the last 40 minutes; request rate to {service} is up "
                    f"{growth_x}x versus the same time last week.")},
        {"tool": "query_vector_db",
         "arguments": {"query": "redis connection pool exhausted timeout runbook"},
         "result": ("Runbook 'CACHE-003: Redis Pool Exhaustion' distinguishes a genuine capacity shortfall "
                    "(pool pinned at 100% with elevated request rate, no recent deploy) from a connection "
                    "leak, and recommends scaling the pool/cache tier for the former.")},
        {"tool": "scale_autoscaling_group",
         "arguments": {"asg_name": f"{service}-cache-tier-asg", "target_capacity": rng.randint(6, 10)},
         "result": f"Scaling {service}-cache-tier-asg to the requested capacity. New pool capacity available within ~90s."},
    ]
    resolution = (
        f"Root cause: sustained {growth_x}x organic traffic growth to {service}, not a deploy or a leak - "
        f"the shared Redis pool was simply undersized for current load. Scaled the cache tier's autoscaling "
        f"group; timeout rate returned to zero."
    )
    return {"injected_logs": logs, "root_cause": root_cause, "tool_trace": tool_trace, "resolution_summary": resolution}


def _gen_disk_full(rng, service, ts, commit):
    pct = round(rng.uniform(96.0, 99.5), 1)
    lines = [
        f"{fmt(ts)} [ERROR] log-forwarder: write failed: no space left on device (/var/log/{service})",
        f"{fmt(ts)} [CRIT] node-exporter: disk usage on /data for {service} at {pct}% (threshold 90%)",
    ]
    logs = "\n".join(lines)
    root_cause = (
        f"Debug-level logging left enabled on {service} after last week's investigation is filling the "
        f"disk faster than the log-rotation policy clears it."
    )
    tool_trace = [
        {"tool": "fetch_service_metrics",
         "arguments": {"metric_name": "disk_utilization", "time_range": "6h"},
         "result": (f"Disk usage on {service}'s volume has climbed steadily from 55% to {pct}% over the "
                    f"last 6 hours; log volume is ~8x the 30-day average.")},
        {"tool": "query_vector_db",
         "arguments": {"query": f"{service} disk full log rotation runbook"},
         "result": ("Runbook 'INFRA-011: Disk Pressure from Logs' recommends scaling storage/capacity "
                    "immediately to avoid an outage, then separately fixing the log level - scaling is a "
                    "mitigation, not the underlying fix.")},
        {"tool": "scale_autoscaling_group",
         "arguments": {"asg_name": f"{service}-asg", "target_capacity": rng.randint(4, 8)},
         "result": f"Scaled {service}-asg to spread log volume across more instances and buy headroom while the logging config is fixed."},
    ]
    resolution = (
        f"Root cause: {service} was left at DEBUG log verbosity after last week's investigation, "
        f"outpacing log rotation. Scaled out {service} as an immediate mitigation to avoid an outage; "
        f"filed a follow-up to revert the log level, which the autoscale alone won't fix."
    )
    return {"injected_logs": logs, "root_cause": root_cause, "tool_trace": tool_trace, "resolution_summary": resolution}


def _gen_cpu_throttle(rng, service, ts, commit):
    throttled_ms = rng.randint(700, 950)
    p99 = rng.randint(2500, 5000)
    lines = [
        f"{fmt(ts)} [WARN] cgroup: CPU throttling detected for {service} "
        f"(throttled_time={throttled_ms}ms/1000ms window)",
        f"{fmt(ts + timedelta(seconds=1))} [ERROR] {service}: request latency p99 = {p99}ms (SLA: 500ms)",
    ]
    logs = "\n".join(lines)
    root_cause = (
        f"Commit {commit} replaced a compiled JSON parser with a pure-Python fallback on {service}'s hot "
        f"path, pushing CPU usage past the container's CPU limit under normal traffic."
    )
    tool_trace = [
        {"tool": "fetch_service_metrics",
         "arguments": {"metric_name": "cpu_throttle_ratio", "time_range": "30m"},
         "result": (f"CPU throttle ratio for {service} jumped from ~2% to {round(throttled_ms / 10)}% "
                    f"right after the last deploy; request volume is flat over the same window.")},
        {"tool": "query_vector_db",
         "arguments": {"query": f"{service} CPU throttling latency runbook"},
         "result": ("Runbook 'PERF-005: CPU Throttling' recommends checking the most recent diff for "
                    "hot-path regressions before requesting a CPU limit increase.")},
        {"tool": "fetch_git_diff",
         "arguments": {"service": service, "commit_hash": commit},
         "result": (f"Commit {commit}: removed the `ujson` dependency and switched "
                    f"`{service}/serializer.py` to the standard-library `json` module 'for portability'.")},
        {"tool": "execute_rollback",
         "arguments": {"service_id": service},
         "result": f"Rollback initiated. {service} now serving from the previous stable revision (pre-{commit})."},
    ]
    resolution = (
        f"Root cause: commit {commit} swapped a compiled JSON serializer for the slower pure-Python "
        f"standard library one on {service}'s hot path, driving CPU usage past its limit. Rolled back; "
        f"p99 latency back under the 500ms SLA."
    )
    return {"injected_logs": logs, "root_cause": root_cause, "tool_trace": tool_trace, "resolution_summary": resolution}


def _gen_bad_deploy_regression(rng, service, ts, commit):
    p99 = rng.randint(1800, 3500)
    err_pct = round(rng.uniform(4.0, 9.0), 1)
    lines = [
        f"{fmt(ts)} [WARN] {service}: request latency p99 = {p99}ms (SLA: 400ms)",
        f"{fmt(ts + timedelta(seconds=2))} [ERROR] {service}: 400 Bad Request rate at {err_pct}% "
        f"(baseline 0.2%), validation_error='missing field: customer_tier'",
    ]
    logs = "\n".join(lines)
    root_cause = (
        f"Commit {commit} added a required `customer_tier` field to {service}'s request schema without a "
        f"default, breaking older clients that don't send it yet."
    )
    tool_trace = [
        {"tool": "fetch_service_metrics",
         "arguments": {"metric_name": "latency_p99", "time_range": "20m"},
         "result": (f"p99 latency for {service} is up to {p99}ms from a ~250ms baseline, and 4xx rate is "
                    f"elevated to {err_pct}%, both starting immediately after the last deploy.")},
        {"tool": "query_vector_db",
         "arguments": {"query": f"{service} elevated 4xx after deploy runbook"},
         "result": ("Runbook 'SVC-021: Post-Deploy Error Spike' recommends diffing the latest commit for "
                    "schema/contract changes before treating this as a capacity issue, since 4xx (client) "
                    "errors rarely respond to scaling.")},
        {"tool": "fetch_git_diff",
         "arguments": {"service": service, "commit_hash": commit},
         "result": (f"Commit {commit}: `customer_tier` added as a required field in the request schema, "
                    f"no default value, no backward-compatible fallback.")},
        {"tool": "execute_rollback",
         "arguments": {"service_id": service},
         "result": f"Rollback initiated. {service} now serving from the previous stable revision (pre-{commit})."},
    ]
    resolution = (
        f"Root cause: commit {commit} made `customer_tier` a required request field with no default, "
        f"breaking clients on older versions. Rolled back {service}; 4xx rate and latency back to baseline."
    )
    return {"injected_logs": logs, "root_cause": root_cause, "tool_trace": tool_trace, "resolution_summary": resolution}


def _gen_traffic_surge(rng, service, ts, commit):
    mult = round(rng.uniform(3.5, 6.0), 1)
    queue_depth = rng.randint(400, 900)
    p99 = rng.randint(2500, 4500)
    lines = [
        f"{fmt(ts)} [WARN] load-balancer: request rate for {service} {mult}x above 7-day baseline",
        f"{fmt(ts + timedelta(seconds=3))} [WARN] autoscaling: {service} at 100% of current desired "
        f"capacity (6/6 healthy targets)",
        f"{fmt(ts + timedelta(seconds=10))} [ERROR] {service}: request queue depth {queue_depth}, "
        f"p99 latency {p99}ms",
    ]
    logs = "\n".join(lines)
    root_cause = (
        f"A scheduled marketing promotion drove a genuine {mult}x traffic surge to {service}; there is no "
        f"code or infrastructure regression, current capacity is simply undersized for the load."
    )
    tool_trace = [
        {"tool": "fetch_service_metrics",
         "arguments": {"metric_name": "request_rate", "time_range": "1h"},
         "result": (f"Request rate to {service} is {mult}x its 7-day baseline for this time of day; error "
                    f"budget is being consumed by queueing, not by failures at the backend.")},
        {"tool": "query_vector_db",
         "arguments": {"query": f"{service} traffic surge capacity runbook"},
         "result": ("Runbook 'CAP-002: Demand Surge' recommends first confirming no recent deploy "
                    "correlates with the surge, then scaling out rather than rolling back.")},
        {"tool": "fetch_git_diff",
         "arguments": {"service": service, "commit_hash": "HEAD"},
         "result": f"No deploys to {service} in the last 24 hours - diff is empty."},
        {"tool": "scale_autoscaling_group",
         "arguments": {"asg_name": f"{service}-asg", "target_capacity": rng.randint(14, 20)},
         "result": f"Scaled {service}-asg to the requested capacity to absorb the surge. Queue depth draining."},
    ]
    resolution = (
        f"Root cause: a marketing promotion drove a real {mult}x demand surge with no code change involved "
        f"- confirmed via an empty git diff over the last 24h. Scaled out {service}'s autoscaling group; "
        f"queue depth and latency back to normal."
    )
    return {"injected_logs": logs, "root_cause": root_cause, "tool_trace": tool_trace, "resolution_summary": resolution}


INCIDENT_GENERATORS = {
    "http_5xx_spike": _gen_http_5xx_spike,
    "oom_kill": _gen_oom_kill,
    "db_deadlock": _gen_db_deadlock,
    "redis_timeout": _gen_redis_timeout,
    "disk_full": _gen_disk_full,
    "cpu_throttle": _gen_cpu_throttle,
    "bad_deploy_regression": _gen_bad_deploy_regression,
    "traffic_surge": _gen_traffic_surge,
}


def build_incident(rng):
    incident_type = rng.choice(list(INCIDENT_GENERATORS.keys()))
    service = rng.choice(SERVICES)
    ts = rand_timestamp(rng)
    commit = rand_commit_hash(rng)
    payload = INCIDENT_GENERATORS[incident_type](rng, service, ts, commit)
    return {
        "incident_id": str(uuid.uuid4()),
        "incident_type": incident_type,
        "service": service,
        "timestamp": fmt(ts),
        **payload,
    }


def _write_jsonl(path, records):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-incidents", type=int, default=600)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    incidents = [build_incident(rng) for _ in range(args.num_incidents)]
    rng.shuffle(incidents)

    n_val = max(1, int(len(incidents) * args.val_fraction))
    val, train = incidents[:n_val], incidents[n_val:]

    os.makedirs(args.output_dir, exist_ok=True)
    _write_jsonl(os.path.join(args.output_dir, "incidents_train.jsonl"), train)
    _write_jsonl(os.path.join(args.output_dir, "incidents_val.jsonl"), val)

    counts = {}
    for inc in incidents:
        counts[inc["incident_type"]] = counts.get(inc["incident_type"], 0) + 1

    print(f"Wrote {len(train)} train / {len(val)} val incidents to {args.output_dir}/")
    print("Class balance:")
    for k, v in sorted(counts.items()):
        print(f"  {k:<24s} {v:>4d}")


if __name__ == "__main__":
    main()
