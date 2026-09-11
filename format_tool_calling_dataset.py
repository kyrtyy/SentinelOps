"""
format_tool_calling_dataset.py

Converts raw incident records (from generate_synthetic_incidents.py)
into multi-turn tool-calling conversations in the schema trl's
SFTTrainer expects for tool-calling SFT:

    {"messages": [...], "tools": [...]}

Each `messages` list alternates system -> user -> (assistant tool_call
-> tool result) * N -> final assistant summary. SFTTrainer passes the
per-example `tools` field straight through to
`tokenizer.apply_chat_template`, so the tool-call syntax always
matches whatever base model you train against (Qwen2.5's <tool_call>
tags, Llama-3.1's own convention, etc.) without any hand-written
templating here.

Usage:
    python format_tool_calling_dataset.py \
        --input data/incidents_train.jsonl --output data/sft_train.jsonl

    python format_tool_calling_dataset.py \
        --input data/incidents_val.jsonl --output data/sft_val.jsonl \
        --check-template Qwen/Qwen2.5-Coder-7B-Instruct
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent.parent))
try:
    from common.tool_schemas import SYSTEM_PROMPT, TOOLS  # noqa: E402
except ImportError:
    from tool_schemas import SYSTEM_PROMPT, TOOLS  # noqa: E402


def build_user_message(incident):
    return (
        f"Incident detected on `{incident['service']}` at {incident['timestamp']}.\n\n"
        f"Raw telemetry / log excerpt:\n```\n{incident['injected_logs'].strip()}\n```\n\n"
        "Diagnose the root cause and take the appropriate remediation action."
    )


def build_messages(incident):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(incident)},
    ]
    for step in incident["tool_trace"]:
        messages.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "type": "function",
                "function": {"name": step["tool"], "arguments": step["arguments"]},
            }],
        })
        messages.append({
            "role": "tool",
            "name": step["tool"],
            "content": step["result"],
        })
    messages.append({"role": "assistant", "content": incident["resolution_summary"]})
    return messages


def convert_file(input_path, output_path):
    n = 0
    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            incident = json.loads(line)
            record = {"messages": build_messages(incident), "tools": TOOLS}
            fout.write(json.dumps(record) + "\n")
            n += 1
    return n


def sanity_check(output_path, model_name, n_samples=2):
    """Render a couple of examples through the real tokenizer's chat
    template so tool-call formatting can be eyeballed before spending
    GPU time on a full training run. Best-effort: skips cleanly if
    transformers isn't installed in the current environment."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("transformers not installed - skipping --check-template sanity check.")
        return

    tok = AutoTokenizer.from_pretrained(model_name)
    with open(output_path) as f:
        for i, line in enumerate(f):
            if i >= n_samples:
                break
            record = json.loads(line)
            rendered = tok.apply_chat_template(
                record["messages"],
                tools=record["tools"],
                tokenize=False,
                add_generation_prompt=False,
            )
            print(f"--- sample {i} rendered with {model_name} ---")
            print(rendered)
            print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--check-template",
        default=None,
        help=("HF model id to sanity-check chat-template rendering against, e.g. "
              "Qwen/Qwen2.5-Coder-7B-Instruct or meta-llama/Llama-3.1-8B-Instruct"),
    )
    args = parser.parse_args()

    n = convert_file(args.input, args.output)
    print(f"Wrote {n} formatted conversations to {args.output}")

    if args.check_template:
        sanity_check(args.output, args.check_template)


if __name__ == "__main__":
    main()
