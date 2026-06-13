"""
experiments/train_lora.py — LoRA Fine-tuning for Delta Agents.

Trains Qwen2.5-7B-Instruct with LoRA for each professional agent:
  - Sentiment Agent: financial news sentiment → 1-10 rating
  - Technical Agent: technical indicators → 1-10 rating
  - Fundamental Agent: financial fundamentals → 1-10 rating

Requires:
  - pip install peft transformers datasets accelerate bitsandbytes

Usage (on GPU machine, e.g. RTX 3060 12GB):
    # Train all 3 agents
    python train_lora.py --all

    # Train individual agent
    python train_lora.py --sentiment
    python train_lora.py --technical
    python train_lora.py --fundamental

    # Quick test (10 steps)
    python train_lora.py --sentiment --quick

    # Custom base model
    python train_lora.py --sentiment --base-model Qwen/Qwen2.5-3B-Instruct
"""

import argparse
import json
import os
from pathlib import Path


def train_lora_agent(
    agent_name: str,
    data_dir: str,
    output_dir: str,
    base_model: str = "Qwen/Qwen2.5-7B-Instruct",
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    learning_rate: float = 2e-4,
    num_epochs: int = 3,
    batch_size: int = 4,
    gradient_accumulation: int = 4,
    max_length: int = 512,
    quick: bool = False,
):
    """Train a LoRA adapter for one agent."""

    print(f"\n{'='*60}")
    print(f"  Training LoRA: {agent_name} Agent")
    print(f"  Base model: {base_model}")
    print(f"  Data: {data_dir}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}")

    # Check data exists
    train_path = Path(data_dir) / "train.json"
    val_path = Path(data_dir) / "val.json"
    if not train_path.exists():
        print(f"  ❌ No training data at {train_path}")
        print(f"  Run: python prepare_lora_data.py --{agent_name}")
        return

    # Import here to allow install-free code review
    try:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            TrainingArguments,
            Trainer,
            DataCollatorForSeq2Seq,
        )
        from peft import LoraConfig, get_peft_model, TaskType
        from datasets import Dataset
    except ImportError as e:
        print(f"  ❌ Missing dependency: {e}")
        print("  Install: pip install peft transformers datasets accelerate bitsandbytes")
        return

    # Check GPU
    if not torch.cuda.is_available():
        print("  ❌ No GPU detected. LoRA training requires CUDA.")
        return

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
    print(f"  GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    # Load tokenizer
    print("  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with 4-bit quantization (fits 7B in 12GB VRAM)
    print("  Loading base model (4-bit quantization)...")
    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # Apply LoRA
    print("  Applying LoRA...")
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load data
    print("  Loading training data...")
    with open(train_path, encoding="utf-8") as f:
        train_data = json.load(f)
    with open(val_path, encoding="utf-8") as f:
        val_data = json.load(f)

    print(f"  Train: {len(train_data)} | Val: {len(val_data)}")

    # Format into chat template
    def format_sample(sample):
        prompt = (
            f"<|im_start|>system\n{sample['instruction']}<|im_end|>\n"
            f"<|im_start|>user\n{sample['input']}<|im_end|>\n"
            f"<|im_start|>assistant\n{sample['output']}<|im_end|>"
        )
        return {"text": prompt}

    train_dataset = Dataset.from_list([format_sample(s) for s in train_data])
    val_dataset = Dataset.from_list([format_sample(s) for s in val_data])

    # Tokenize
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    train_dataset = train_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    val_dataset = val_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

    # Training arguments
    if quick:
        num_epochs = 1
        max_steps = 10

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50 if not quick else 5,
        save_strategy="steps",
        save_steps=100 if not quick else 10,
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
    )

    if quick:
        training_args.max_steps = 10

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        max_length=max_length,
    )

    # Train
    print("  Starting training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    trainer.train()

    # Save LoRA adapter
    print(f"  Saving LoRA adapter to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save training config
    config = {
        "agent_name": agent_name,
        "base_model": base_model,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "quick_mode": quick,
    }
    with open(Path(output_dir) / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"  ✅ {agent_name} LoRA training complete!")
    print(f"  Adapter saved: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train LoRA adapters for Delta Agents")
    parser.add_argument("--all", action="store_true", help="Train all 3 agents")
    parser.add_argument("--sentiment", action="store_true")
    parser.add_argument("--technical", action="store_true")
    parser.add_argument("--fundamental", action="store_true")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--data-dir", default="data/lora")
    parser.add_argument("--output-dir", default="agents/lora")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--quick", action="store_true", help="Quick test: 10 steps only")
    args = parser.parse_args()

    if not any([args.all, args.sentiment, args.technical, args.fundamental]):
        args.all = True

    agents = []
    if args.all or args.sentiment:
        agents.append(("sentiment", f"{args.data_dir}/sentiment", f"{args.output_dir}/sentiment"))
    if args.all or args.technical:
        agents.append(("technical", f"{args.data_dir}/technical", f"{args.output_dir}/technical"))
    if args.all or args.fundamental:
        agents.append(("fundamental", f"{args.data_dir}/fundamental", f"{args.output_dir}/fundamental"))

    for name, data, output in agents:
        train_lora_agent(
            agent_name=name,
            data_dir=data,
            output_dir=output,
            base_model=args.base_model,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            learning_rate=args.lr,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            quick=args.quick,
        )

    print("\n✅ All training complete!")
    print("Adapters saved in agents/lora/")
    print("Next: python step2_scoring.py --group B")


if __name__ == "__main__":
    main()
