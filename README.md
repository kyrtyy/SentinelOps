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
- [x] **Evaluation benchmark** - Automated SRE scoring, LLM-as-a-Judge (`evaluate_benchmark.py`), scorecard generation
- [x] **Infrastructure (Terraform)** - Modular AWS architecture: VPC, EC2 Spot GPU (A10G), RDS pgvector, EventBridge + Lambda

## Architecture at a glance

- **Base model:** `Qwen/Qwen2.5-Coder-7B-Instruct`
- **Fine-tuning:** LoRA via PEFT + TRL's `SFTTrainer`, `bfloat16`, PyTorch native SDPA, distributed with `accelerate` DDP
- **Serving:** OpenAI-compatible FastAPI server (`serve_api.py`) or vLLM
- **RAG:** pgvector-backed runbook retrieval (`data/runbooks.jsonl`)
- **Orchestration:** Multi-turn ReAct agent loop (`agent_orchestrator.py`), event-driven off CloudWatch alarms
- **Evaluation:** LLM-as-a-Judge and automated ground-truth scoring (`evaluate_benchmark.py`)
- **Cloud Infrastructure:** Terraform modules under `terraform/` for automated AWS deployment

## Pipeline & Commands

### 1. Data Ingestion & Generation
```bash
# Generate synthetic failure corpus
python generate_synthetic_incidents.py --num-incidents 600 --output-dir data --seed 13

# Blend open-source real-world SRE trajectories
python ingest_open_sre_data.py --output-dir data \
    --synthetic-train data/incidents_train.jsonl \
    --synthetic-val data/incidents_val.jsonl

# Ingest runbooks & post-mortems for RAG
python ingest_postmortems.py --output data/runbooks.jsonl
```

### 2. Distributed Training & Checkpoint Merge
```bash
# Launch multi-GPU LoRA SFT training
accelerate launch --multi_gpu --mixed_precision bf16 train_ddp.py --config train_lora_config.yaml

# Merge adapter into base weights
python merge_lora.py --base-model Qwen/Qwen2.5-Coder-7B-Instruct --adapter outputs/sentinelops-lora --output outputs/sentinelops-merged
```

### 3. Model Serving & Autonomous ReAct Agent
```bash
# Start OpenAI-compatible API server
python serve_api.py --model-path outputs/sentinelops-merged --port 8000

# Run autonomous agent loop against test incidents
python agent_orchestrator.py --api-base http://localhost:8000/v1 --scenario db_deadlock
```

### 4. Evaluation Benchmark (LLM-as-a-Judge)
```bash
python evaluate_benchmark.py --api-base http://localhost:8000/v1 --output benchmark_scorecard.md
```

### 5. AWS Cloud Infrastructure Deployment (Terraform)
```bash
cd terraform
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your credentials, then run:
./deploy.sh
```

## Repository layout

```text
sentinelops/
├── tool_schemas.py                 # Shared tool JSON-schemas + system prompt
├── generate_synthetic_incidents.py # 8-fault synthetic incident generator
├── ingest_open_sre_data.py         # Ingests OpenSRE & real-world outage trajectories
├── ingest_postmortems.py           # Ingests post-mortems for RAG vector DB
├── format_tool_calling_dataset.py  # Multi-turn chat & tool-calling formatter
├── train_ddp.py                    # Distributed LoRA SFT trainer
├── train_lora_config.yaml          # Hyperparameters and dataset paths
├── merge_lora.py                   # LoRA adapter checkpoint merge utility
├── quantize_awq.py                 # AWQ INT4 quantization utility
├── serve_api.py                    # Lightweight OpenAI-compatible FastAPI model server
├── test_agent_inference.py         # Standalone tool-calling inference tester
├── agent_orchestrator.py           # Autonomous multi-turn ReAct agent loop
├── evaluate_benchmark.py           # Evaluation benchmark & LLM-as-a-Judge scorer
├── requirements.txt                # Python dependencies
└── terraform/                      # AWS Cloud Infrastructure as Code
    ├── main.tf                     # Root Terraform config
    ├── variables.tf                # AWS region, DB, and instance variables
    ├── outputs.tf                  # IP addresses, endpoints, and ARNs
    ├── deploy.sh                   # One-click deployment script
    └── modules/
        ├── vpc/                    # Multi-AZ VPC and subnets
        ├── ec2_spot/               # EC2 Spot GPU instance (NVIDIA A10G 24GB)
        ├── rds_pgvector/           # RDS PostgreSQL + pgvector runbook store
        └── eventbridge_lambda/     # CloudWatch Alarm EventBridge rule & Lambda trigger
```
