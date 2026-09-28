"""
DPO (Direct Preference Optimization) Fine-Tuning.
Trains the model to prefer good JSON outputs over bad ones using
preference pairs (chosen vs rejected).

This step runs AFTER SFT/LoRA to align the model's output style.
Eliminates reward hacking present in traditional RLHF.
"""

import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import PeftModel, LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOTrainer, DPOConfig


SFT_MODEL_PATH = "results/lora-json-extractor"
BASE_MODEL = "microsoft/phi-3.5-mini-instruct"
OUTPUT_DIR = "results/dpo-json-extractor"

# DPO preference data — (prompt, chosen_response, rejected_response)
DPO_DATA = [
    {
        "prompt": "Extract JSON: Alice Johnson, 28, Senior Data Scientist at TechCorp. Email: alice@tech.com",
        "chosen": '{"name": "Alice Johnson", "age": 28, "role": "Senior Data Scientist", "company": "TechCorp", "email": "alice@tech.com"}',
        "rejected": 'Sure! Here is the extracted JSON:\n```json\n{"name": "Alice Johnson", "age": 28}\n```\nHope that helps!',
    },
    {
        "prompt": "Extract JSON: Invoice #999, 2x Laptop @ $899 each, total $1798, due 2025-04-01",
        "chosen": '{"invoice_id": "999", "items": [{"name": "Laptop", "qty": 2, "unit_price": 899}], "total": 1798, "due_date": "2025-04-01"}',
        "rejected": "The invoice contains a laptop order. The total is $1,798 and it's due April 1st.",
    },
    {
        "prompt": "Extract JSON: Flight BA234 departs LHR at 08:15, arrives JFK at 11:30, date 2025-06-10",
        "chosen": '{"flight": "BA234", "origin": "LHR", "destination": "JFK", "departure": "08:15", "arrival": "11:30", "date": "2025-06-10"}',
        "rejected": '{"flight": "BA234", "route": "LHR to JFK"}',  # Incomplete
    },
]


def prepare_dpo_dataset() -> Dataset:
    """Load DPO preference pairs."""
    data_path = "src/data/dpo_data.jsonl"
    rows = []
    if os.path.exists(data_path):
        with open(data_path) as f:
            for line in f:
                rows.append(json.loads(line))
    else:
        rows = DPO_DATA
        print(f"[DPO] Using {len(rows)} sample preference pairs")

    return Dataset.from_list(rows)


def train_dpo():
    print("[DPO] Loading SFT-tuned model from:", SFT_MODEL_PATH)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Load base + SFT adapter as the policy model
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token

    model = PeftModel.from_pretrained(base, SFT_MODEL_PATH)

    dataset = prepare_dpo_dataset()

    dpo_config = DPOConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        bf16=True,
        beta=0.1,          # KL divergence penalty coefficient
        logging_steps=5,
        save_strategy="epoch",
        report_to="none",
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,   # Use implicit reference (frozen copy of model)
        args=dpo_config,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    print("[DPO] Starting preference training...")
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    print(f"[DPO] ✅ DPO-tuned adapter saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    train_dpo()
