"""
LoRA / QLoRA Supervised Fine-Tuning for JSON Extraction.
Task: Given unstructured text, extract structured JSON.

Uses:
  - Hugging Face PEFT for LoRA adapters
  - bitsandbytes for 4-bit quantization (QLoRA)
  - trl SFTTrainer for training
  - Saves adapter weights only (~20MB vs ~7GB for full model)
"""

from dataclasses import dataclass, field
from typing import Optional
import json
import os

import torch
from datasets import Dataset, load_dataset
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer


# ---- Config ----
BASE_MODEL = "microsoft/phi-3.5-mini-instruct"  # 3.8B — runs on 16GB GPU
OUTPUT_DIR = "results/lora-json-extractor"
LORA_RANK = 16
LORA_ALPHA = 32
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
MAX_SEQ_LEN = 512
BATCH_SIZE = 4
GRAD_ACCUM = 4
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
USE_4BIT = True  # QLoRA — set False if you have 40GB+ GPU


SYSTEM_PROMPT = """You are a precise JSON extraction assistant.
Extract structured information from text and return ONLY valid JSON.
No explanation, no markdown fences — raw JSON only."""

# ---- Training Data ----
SAMPLE_DATA = [
    {
        "text": "John Smith, age 34, works at Acme Corp as a Software Engineer. Email: john@acme.com",
        "output": '{"name": "John Smith", "age": 34, "company": "Acme Corp", "role": "Software Engineer", "email": "john@acme.com"}',
    },
    {
        "text": "Order #4821: 3x Widget Pro @ $12.99 each. Shipped to 42 Oak Street, Boston MA 02101. Status: delivered.",
        "output": '{"order_id": "4821", "items": [{"name": "Widget Pro", "qty": 3, "unit_price": 12.99}], "address": "42 Oak Street, Boston MA 02101", "status": "delivered"}',
    },
    {
        "text": "Meeting scheduled for March 15 2025 at 2pm EST with Alice (alice@corp.com) and Bob (bob@corp.com) to discuss Q2 roadmap.",
        "output": '{"date": "2025-03-15", "time": "14:00 EST", "attendees": [{"name": "Alice", "email": "alice@corp.com"}, {"name": "Bob", "email": "bob@corp.com"}], "topic": "Q2 roadmap"}',
    },
]


def format_prompt(example):
    """Format as instruction → response for SFT."""
    prompt = f"<|system|>{SYSTEM_PROMPT}<|end|>\n<|user|>{example['text']}<|end|>\n<|assistant|>{example['output']}<|end|>"
    return {"text": prompt}


def load_training_data() -> Dataset:
    """Load from file if present, else use sample data."""
    data_path = "src/data/training_data.jsonl"
    if os.path.exists(data_path):
        rows = []
        with open(data_path) as f:
            for line in f:
                rows.append(json.loads(line))
        dataset = Dataset.from_list(rows)
    else:
        dataset = Dataset.from_list(SAMPLE_DATA)
        print(f"[SFT] Using {len(dataset)} sample examples (add more to src/data/training_data.jsonl)")

    return dataset.map(format_prompt)


def get_bnb_config():
    """4-bit quantization config for QLoRA."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def train():
    print(f"[SFT] Loading base model: {BASE_MODEL}")

    # Load model
    model_kwargs = {}
    if USE_4BIT:
        model_kwargs["quantization_config"] = get_bnb_config()
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, **model_kwargs
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    # Prepare for k-bit training (QLoRA)
    if USE_4BIT:
        model = prepare_model_for_kbit_training(model)

    # Apply LoRA adapters
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load data
    train_dataset = load_training_data()

    # Training args
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        tokenizer=tokenizer,
    )

    print("[SFT] Starting training...")
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"[SFT] ✅ Adapter saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    train()
