"""
Before/After Evaluation: Compares base model vs LoRA vs DPO model
on JSON extraction quality.

Metrics:
  - JSON validity rate (is the output parseable?)
  - Field precision (how many expected fields are present?)
  - Purity rate (is there any extra text beyond JSON?)
  - Inference time
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from rich.console import Console
from rich.table import Table


console = Console()

BASE_MODEL = "microsoft/phi-3.5-mini-instruct"
LORA_PATH = "results/lora-json-extractor"
DPO_PATH = "results/dpo-json-extractor"

EVAL_CASES = [
    {
        "input": "Dr. Maria Garcia, 45, Cardiologist at City Hospital. Phone: 555-0123",
        "expected_fields": ["name", "age", "role", "hospital", "phone"],
    },
    {
        "input": "Package delivered: Order 7721, 1x Headphones $149.99, 2x USB-C Cable $9.99 each, to 10 Main St, Chicago IL",
        "expected_fields": ["order_id", "items", "address"],
    },
    {
        "input": "PR #482 opened by dev-alex, title: 'Fix null pointer in auth module', labels: bug, urgent",
        "expected_fields": ["pr_number", "author", "title", "labels"],
    },
]

SYSTEM = "You are a JSON extraction assistant. Return ONLY valid JSON. No explanation."


def load_model(model_path: Optional[str] = None, is_peft: bool = False):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path or BASE_MODEL)
    if is_peft and model_path:
        model = PeftModel.from_pretrained(base, model_path)
    else:
        model = base
    return model, tokenizer


def infer(model, tokenizer, text: str) -> tuple[str, float]:
    prompt = f"<|system|>{SYSTEM}<|end|>\n<|user|>{text}<|end|>\n<|assistant|>"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    start = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=256, temperature=0.1, do_sample=True)
    elapsed = (time.perf_counter() - start) * 1000
    generated = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return generated.strip(), elapsed


def evaluate_output(output: str, expected_fields: List[str]) -> Dict:
    # JSON validity
    try:
        parsed = json.loads(output)
        valid = True
    except Exception:
        parsed = {}
        valid = False

    # Field precision
    present = sum(1 for f in expected_fields if f in parsed)
    precision = present / len(expected_fields)

    # Purity (no extra text)
    pure = valid and output.strip().startswith("{")

    return {"valid": valid, "field_precision": precision, "pure": pure}


def run_evaluation():
    results = {}
    configs = [
        ("Base (no fine-tuning)", None, False),
        ("LoRA SFT", LORA_PATH, True),
        ("LoRA + DPO", DPO_PATH, True),
    ]

    for name, path, is_peft in configs:
        if path and not Path(path).exists():
            console.print(f"[yellow]Skipping {name}: {path} not found[/yellow]")
            continue

        console.print(f"\n[cyan]Evaluating: {name}[/cyan]")
        model, tokenizer = load_model(path, is_peft)

        scores = {"valid": [], "field_precision": [], "pure": [], "latency_ms": []}
        for case in EVAL_CASES:
            output, lat = infer(model, tokenizer, case["input"])
            metrics = evaluate_output(output, case["expected_fields"])
            scores["valid"].append(metrics["valid"])
            scores["field_precision"].append(metrics["field_precision"])
            scores["pure"].append(metrics["pure"])
            scores["latency_ms"].append(lat)

        results[name] = {
            "json_validity": sum(scores["valid"]) / len(scores["valid"]) * 100,
            "field_precision": sum(scores["field_precision"]) / len(scores["field_precision"]) * 100,
            "purity_rate": sum(scores["pure"]) / len(scores["pure"]) * 100,
            "avg_latency_ms": sum(scores["latency_ms"]) / len(scores["latency_ms"]),
        }

        del model  # Free GPU memory

    # Print results table
    table = Table(title="Before vs After Fine-Tuning")
    table.add_column("Model", style="cyan")
    table.add_column("JSON Valid %", justify="right")
    table.add_column("Field Precision %", justify="right")
    table.add_column("Purity %", justify="right")
    table.add_column("Latency (ms)", justify="right")

    for name, r in results.items():
        table.add_row(
            name,
            f"{r['json_validity']:.0f}%",
            f"{r['field_precision']:.0f}%",
            f"{r['purity_rate']:.0f}%",
            f"{r['avg_latency_ms']:.0f}",
        )

    console.print(table)

    # Save results
    Path("results").mkdir(exist_ok=True)
    with open("results/eval_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    console.print("[green]Saved to results/eval_comparison.json[/green]")


if __name__ == "__main__":
    run_evaluation()
