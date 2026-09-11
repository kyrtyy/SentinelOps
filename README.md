# SentinelOps

Autonomous SRE incident-response agent - a portfolio/showcase project (not for
commercial use, no production traffic). Fine-tunes an open-source 7-8B model
for structured incident triage and tool-calling, quantizes it, serves it with
vLLM on AWS, and wraps it in a RAG-backed ReAct agent evaluated against a
larger model acting as judge.

## Status

- [x] **Training pipeline** - open-source & synthetic incident dataset, LoRA SFT (PyTorch DDP)
- [x] **Model serving** - OpenAI-compatible server (`serve_api.py`), AWQ INT4 support
- [x] **Agent orchestrator** - Autonomous multi-turn ReAct loop (`agent_orchestrator.py`), tool execution & RAG runbooks
- [ ] Infrastructure (Terraform: VPC, EC2 spot, RDS+pgvector, Lambda, EventBridge)
- [ ] Evaluation (Bedrock Claude 3.5 Sonnet as judge, latency/throughput benchmarks)

## Architecture at a glance

- **Base model:** `Qwen/Qwen2.5-Coder-7B-Instruct` or `meta-llama/Llama-3.1-8B-Instruct`
- **Fine-tuning:** LoRA via PEFT + TRL's `SFTTrainer`, `bfloat16`, FlashAttention-2,
  distributed with `accelerate` DDP across 2x RTX 4090 (24GB)
- **Quantization:** AWQ INT4 for vLLM serving on a single A10G (`g5.xlarge`)
- **Serving:** vLLM, OpenAI-compatible endpoint
- **RAG:** pgvector-backed runbook retrieval
- **Orchestration:** ReAct agent loop, event-driven off CloudWatch alarms
- **Evaluation:** Amazon Bedrock (Claude 3.5 Sonnet) as LLM-judge

### Design notes

A couple of deliberate departures from the "obvious" heavyweight choice,
worth calling out since this is a showcase project:

- **DDP instead of DeepSpeed ZeRO.** LoRA's trainable-parameter footprint is
  tiny relative to the frozen base model, which comfortably fits on a single
  24GB GPU in bf16. ZeRO-2/3 exists to shard models that don't fit in one
  GPU's memory - using it here would add real complexity for no benefit.
- **AWQ INT4 instead of FP8 for serving.** The target serving GPU (A10G,
  Ampere) has no FP8 tensor cores; FP8 needs Hopper or Ada Lovelace. AWQ is
  the correct - and only working - choice for this hardware.

## Training pipeline

### 1. Generate or Ingest SRE Incident Trajectories

You can generate synthetic incidents, ingest open-source real-world SRE benchmarks (e.g. Hugging Face `quantranger/opensre-incident-trajectories` / curated real-world outages from AWS, Cloudflare, Slack, GitHub, Datadog), or blend both:

**Option A: Generate Synthetic Incidents (Offline, zero API dependencies)**
```bash
python generate_synthetic_incidents.py \
    --num-incidents 600 \
    --output-dir data \
    --seed 13
```

**Option B: Ingest & Blend Open-Source SRE Trajectories (Real + Synthetic)**
```bash
# Ingests curated real-world failure cases and blends with synthetic traces
python ingest_open_sre_data.py \
    --output-dir data \
    --synthetic-train data/incidents_train.jsonl \
    --synthetic-val data/incidents_val.jsonl \
    --include-hf
```

### 2. Format into Tool-Calling Conversations

```bash
python format_tool_calling_dataset.py \
    --input data/incidents_train.jsonl --output data/sft_train.jsonl

python format_tool_calling_dataset.py \
    --input data/incidents_val.jsonl --output data/sft_val.jsonl \
    --check-template Qwen/Qwen2.5-Coder-7B-Instruct
```

### 3. Ingest Runbooks & Post-Mortems for RAG

Export structured post-mortems for indexing into pgvector (`query_vector_db`):
```bash
python ingest_postmortems.py --output data/runbooks.jsonl
```

### 4. Fine-tune

```bash
pip install -r requirements.txt
accelerate launch --config_file accelerate_config.yaml \
    train_ddp.py --config train_lora_config.yaml
```

Edit `train_lora_config.yaml` to switch base model, LoRA rank, or any
training hyperparameter - `train_ddp.py` itself shouldn't need touching.

### 5. Merge the Adapter

```bash
python merge_lora.py \
    --base-model Qwen/Qwen2.5-Coder-7B-Instruct \
    --adapter outputs/sentinelops-lora \
    --output outputs/sentinelops-merged
```

The merged checkpoint in `outputs/sentinelops-merged` is the input to the
next phase (AWQ quantization + vLLM serving).

## Repository layout

```text
sentinelops/
├── tool_schemas.py                 # Shared tool JSON-schemas + system prompt
├── generate_synthetic_incidents.py # 8-fault synthetic incident generator
├── ingest_open_sre_data.py         # Ingests OpenSRE & real-world outage trajectories
├── ingest_postmortems.py           # Ingests post-mortems for RAG vector DB
├── format_tool_calling_dataset.py  # Multi-turn chat & tool-calling formatter
├── accelerate_config.yaml          # Multi-GPU DDP training configuration
├── train_lora_config.yaml          # Hyperparameters and dataset paths
├── train_ddp.py                    # Distributed LoRA SFT trainer
├── merge_lora.py                   # LoRA adapter checkpoint merge utility
└── requirements.txt
```
