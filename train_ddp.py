"""
train_ddp.py

LoRA SFT of an open-source 7B/8B instruct model for SentinelOps
incident-triage tool-calling, distributed across multiple GPUs via
accelerate (DDP) rather than DeepSpeed ZeRO - LoRA's trainable-
parameter footprint is tiny relative to the frozen base model, which
comfortably fits on a single 24GB GPU in bf16, so sharding buys
nothing at this scale.

Launch with:
    accelerate launch --config_file configs/accelerate_config.yaml \
        training/train_ddp.py --config configs/train_lora_config.yaml

Requires trl>=0.19 for native {"messages": ..., "tools": ...}
tool-calling SFT support - install the latest release
(`pip install -U trl`), since assistant_only_loss has had several
bugfixes since it landed. Qwen2.5 and Llama-3/3.1 chat templates are
auto-patched by TRL when assistant_only_loss=True; a different base
model may need a custom {% generation %} template first - see
https://huggingface.co/docs/trl/chat_templates.
"""
import argparse
import os
import sys
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent.parent))
try:
    from common.tool_schemas import TOOLS  # noqa: E402
except ImportError:
    from tool_schemas import TOOLS  # noqa: E402


def load_yaml_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def attach_tools_column(dataset):
    """format_tool_calling_dataset.py already writes a `tools` column;
    this re-asserts it so the script still works against an older
    dataset file that predates that field."""
    if "tools" in dataset.column_names:
        return dataset
    return dataset.add_column("tools", [TOOLS] * len(dataset))


def build_sft_config(cfg, eval_dataset):
    import dataclasses
    import inspect

    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    seq_len = model_cfg.get("max_seq_length", 4096)

    # Base training arguments
    kwargs = {
        "output_dir": train_cfg["output_dir"],
        "num_train_epochs": train_cfg.get("num_train_epochs", 3),
        "per_device_train_batch_size": train_cfg.get("per_device_train_batch_size", 2),
        "gradient_accumulation_steps": train_cfg.get("gradient_accumulation_steps", 8),
        "learning_rate": float(train_cfg.get("learning_rate", 2e-4)),
        "lr_scheduler_type": train_cfg.get("lr_scheduler_type", "cosine"),
        "warmup_ratio": float(train_cfg.get("warmup_ratio", 0.03)),
        "weight_decay": float(train_cfg.get("weight_decay", 0.01)),
        "bf16": bool(train_cfg.get("bf16", True)),
        "gradient_checkpointing": bool(train_cfg.get("gradient_checkpointing", True)),
        "logging_steps": int(train_cfg.get("logging_steps", 10)),
        "eval_strategy": (train_cfg.get("eval_strategy", "steps") if eval_dataset is not None else "no"),
        "eval_steps": int(train_cfg.get("eval_steps", 50)),
        "save_strategy": train_cfg.get("save_strategy", "steps"),
        "save_steps": int(train_cfg.get("save_steps", 50)),
        "save_total_limit": int(train_cfg.get("save_total_limit", 3)),
        "seed": int(train_cfg.get("seed", 13)),
        "assistant_only_loss": bool(train_cfg.get("assistant_only_loss", True)),
        "packing": bool(train_cfg.get("packing", False)),
        "report_to": train_cfg.get("report_to", "none"),
        "run_name": train_cfg.get("run_name"),
        "max_seq_length": seq_len,
        "max_length": seq_len,
    }

    # Inspect valid fields for the installed SFTConfig version
    valid_fields = set()
    if dataclasses.is_dataclass(SFTConfig):
        valid_fields = {f.name for f in dataclasses.fields(SFTConfig)}
    else:
        try:
            sig = inspect.signature(SFTConfig.__init__)
            valid_fields = set(sig.parameters.keys())
        except Exception:
            pass

    if valid_fields:
        filtered = {k: v for k, v in kwargs.items() if k in valid_fields}
        # Deduplicate max_seq_length / max_length
        if "max_seq_length" in filtered and "max_length" in filtered:
            if "max_length" in valid_fields and "max_seq_length" not in valid_fields:
                filtered.pop("max_seq_length", None)
            else:
                filtered.pop("max_length", None)
        config = SFTConfig(**filtered)
    else:
        try:
            config = SFTConfig(**kwargs)
        except TypeError:
            # Fallback minimal init
            config = SFTConfig(output_dir=train_cfg["output_dir"])

    # Ensure all attributes are assigned on the instance
    for k, v in kwargs.items():
        if k in ("max_length", "max_seq_length"):
            continue
        try:
            setattr(config, k, v)
        except Exception:
            pass

    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to train_lora_config.yaml")
    args = parser.parse_args()
    cfg = load_yaml_config(args.config)

    model_cfg = cfg["model"]
    lora_cfg = cfg["lora"]
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    wandb_cfg = cfg.get("wandb", {})

    if wandb_cfg.get("project"):
        os.environ.setdefault("WANDB_PROJECT", wandb_cfg["project"])

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name_or_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    attn_impl = model_cfg.get("attn_implementation", "flash_attention_2")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_cfg["name_or_path"],
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_impl,
        )
    except Exception as e:
        if attn_impl == "flash_attention_2":
            print(f"Warning: flash_attention_2 failed to load ({e}). Falling back to native PyTorch 'sdpa'...")
            model = AutoModelForCausalLM.from_pretrained(
                model_cfg["name_or_path"],
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
            )
        else:
            raise

    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg.get("bias", "none"),
        task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
    )

    train_dataset = attach_tools_column(
        load_dataset("json", data_files=data_cfg["train_file"], split="train")
    )
    eval_dataset = None
    if data_cfg.get("eval_file"):
        eval_dataset = attach_tools_column(
            load_dataset("json", data_files=data_cfg["eval_file"], split="train")
        )

    sft_config = build_sft_config(cfg, eval_dataset)

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(train_cfg["output_dir"])
    tokenizer.save_pretrained(train_cfg["output_dir"])
    print(f"LoRA adapter saved to {train_cfg['output_dir']}")


if __name__ == "__main__":
    main()
