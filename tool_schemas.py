"""
common/tool_schemas.py

Single source of truth for SentinelOps' tool definitions and system
prompt. Imported by the dataset formatter at training time and, in
the agent-orchestrator phase, by agent/tools.py at serving time - so
the schema the model is trained on is byte-for-byte what it's served
with.

Tool JSON schemas are passed to `tokenizer.apply_chat_template(...,
tools=TOOLS)` rather than hand-embedded into the system prompt text:
each base model's chat template already knows how to render its own
tool-call syntax (Qwen2.5's <tool_call> tags, Llama-3.1's convention,
etc.), so let it do that instead of duplicating/fighting it here.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "fetch_service_metrics",
            "description": (
                "Fetch a time-series metric (error rate, latency, CPU, memory, "
                "disk, connection-pool usage, request rate, etc.) for a service "
                "over a recent time window."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "metric_name": {
                        "type": "string",
                        "description": (
                            "e.g. 'http_5xx_rate', 'memory_utilization', "
                            "'cpu_throttle_ratio', 'db_deadlock_count', "
                            "'redis_connection_pool_usage', 'disk_utilization', "
                            "'request_rate', 'latency_p99'."
                        ),
                    },
                    "time_range": {
                        "type": "string",
                        "description": "Lookback window, e.g. '15m', '30m', '1h', '6h'.",
                    },
                },
                "required": ["metric_name", "time_range"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_vector_db",
            "description": "Semantic search over indexed runbooks, past post-mortems, and architecture docs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural-language search query."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_git_diff",
            "description": (
                "Fetch the diff for a commit deployed to a service, to check for "
                "recent code changes that could explain an incident. Use "
                "commit_hash='HEAD' to check the most recent deploy."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "service": {"type": "string"},
                    "commit_hash": {"type": "string"},
                },
                "required": ["service", "commit_hash"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_rollback",
            "description": "Roll a service back to its previous stable deployed revision.",
            "parameters": {
                "type": "object",
                "properties": {"service_id": {"type": "string"}},
                "required": ["service_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scale_autoscaling_group",
            "description": "Change the desired capacity of an autoscaling group.",
            "parameters": {
                "type": "object",
                "properties": {
                    "asg_name": {"type": "string"},
                    "target_capacity": {
                        "type": "integer",
                        "description": "New desired instance/pod count.",
                    },
                },
                "required": ["asg_name", "target_capacity"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_logs",
            "description": "Fetch application or container log stream for a specific service or pod with optional filtering.",
            "parameters": {
                "type": "object",
                "properties": {
                    "service": {"type": "string", "description": "Target service or pod name."},
                    "filter_pattern": {"type": "string", "description": "Regex or substring filter, e.g. 'ERROR', 'FATAL', 'panic'."},
                    "limit": {"type": "integer", "description": "Number of log lines to retrieve (default: 50)."},
                },
                "required": ["service"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_pod",
            "description": "Inspect Kubernetes pod/container status, restart counts, resource limits, and event history.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pod_name": {"type": "string", "description": "Pod identifier or prefix."},
                    "namespace": {"type": "string", "description": "Kubernetes namespace (default: 'default')."},
                },
                "required": ["pod_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_traces",
            "description": "Query distributed tracing spans (e.g. OpenTelemetry / Jaeger) for high-latency or errored request traces.",
            "parameters": {
                "type": "object",
                "properties": {
                    "service": {"type": "string"},
                    "min_duration_ms": {"type": "integer", "description": "Filter traces slower than this threshold in ms."},
                    "status_code": {"type": "string", "description": "Filter by HTTP/gRPC status, e.g. '500', 'ERROR'."},
                },
                "required": ["service"],
            },
        },
    },
]

SYSTEM_PROMPT = (
    "You are SentinelOps, an autonomous SRE incident-response agent. You are "
    "given raw telemetry and log excerpts for a production incident. "
    "Investigate using the available tools, determine the root cause, and take "
    "the correct remediation action.\n\n"
    "Always check metrics, logs, traces, and the runbook knowledge base before acting. Only "
    "call execute_rollback or scale_autoscaling_group once you have specific "
    "evidence pointing to that fix - check fetch_git_diff before assuming a "
    "deploy caused the incident, since some incidents (e.g. organic traffic "
    "growth) have no code change behind them at all. Respond with a tool call "
    "when you need more information or need to act, and respond with a final "
    "plain-text summary once the incident is resolved."
)
