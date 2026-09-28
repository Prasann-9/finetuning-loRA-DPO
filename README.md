# 🧠 Fine-Tuning with LoRA & DPO

## What Is It?
Fine-tunes a 3.8B parameter language model to precisely extract structured JSON from unstructured text. Uses **LoRA** for memory-efficient training (adapters only, not full weights), then **DPO** for preference alignment (teach it to prefer clean JSON over verbose explanations).

---

## Industrial Applications
| Domain | Use Case |
|---|---|
| Healthcare | Extract structured patient data from clinical notes |
| Finance | Parse transaction data from emails/PDFs |
| Legal | Extract parties, dates, clauses from contracts |
| DevOps | Parse incident reports into structured tickets |
| E-commerce | Extract product specs from unstructured listings |

---

## Key Technical Concepts

### LoRA (Low-Rank Adaptation)
Instead of updating all 3.8 billion weights (which requires 28GB+ GPU memory), LoRA inserts small trainable "adapter" matrices into the attention layers. You train only ~0.5% of parameters. The result: same fine-tuning quality, but fits on a single 16GB GPU.

```
Full Fine-Tuning:  W_new = W_original + ΔW           (massive ΔW)
LoRA:              W_new = W_original + A × B         (A and B are tiny matrices, rank=16)
```

### QLoRA (Quantized LoRA)
Loads the base model in 4-bit precision (using NF4 quantization via bitsandbytes), reducing memory from ~8GB to ~3GB. LoRA adapters are still trained in full precision. Makes 7B+ models trainable on consumer GPUs.

### DPO (Direct Preference Optimization)
After SFT, the model might say "Sure! Here's the JSON:" before the actual output. DPO fixes this by training on pairs:
- **Chosen**: `{"name": "Alice", "age": 28}` ← clean JSON
- **Rejected**: `Sure! Here is the JSON:\n\`\`\`json\n{"name": "Alice"}\n\`\`\`` ← verbose

No reward model needed. The model directly learns to prefer the chosen format.

### Before vs After Metrics
- **JSON Validity Rate**: % of outputs that are parseable JSON
- **Field Precision**: % of expected fields correctly extracted
- **Purity Rate**: % of outputs containing ONLY JSON (no extra text)

---

## File Structure
```
04-finetuning-lora-dpo/
├── src/
│   ├── training/
│   │   ├── sft_lora.py      # Step 1: LoRA supervised fine-tuning
│   │   └── dpo_train.py     # Step 2: DPO preference alignment
│   ├── evaluation/
│   │   └── compare.py       # Before/after metrics comparison
│   └── data/
│       ├── training_data.jsonl  # SFT examples (add yours here)
│       └── dpo_data.jsonl       # DPO preference pairs
└── requirements.txt
```

---

# ▶️ How to Run

## Requirements
- Python 3.11+
- NVIDIA GPU with 16GB+ VRAM (RTX 3090, 4090, A100)
- CUDA 12.x

## Setup
```bash
cd 04-finetuning-lora-dpo
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Step 1 — SFT with LoRA
```bash
python -m src.training.sft_lora
# Saves adapter to results/lora-json-extractor/
# Prints: "trainable params: X || all params: Y || trainable%: 0.5"
```

## Step 2 — DPO Preference Alignment
```bash
python -m src.training.dpo_train
# Loads SFT adapter, runs DPO, saves to results/dpo-json-extractor/
```

## Step 3 — Compare Before/After
```bash
python -m src.evaluation.compare
# Prints table: base model vs LoRA vs LoRA+DPO
# Saves results/eval_comparison.json
```

## Add Your Own Training Data
```bash
# SFT data format (one JSON object per line):
echo '{"text": "raw text here", "output": "{\"key\": \"value\"}"}' >> src/data/training_data.jsonl

# DPO data format:
echo '{"prompt": "...", "chosen": "{clean json}", "rejected": "verbose response"}' >> src/data/dpo_data.jsonl
```

---

# 📐 Strategy

## Why This Task?
JSON extraction is a perfect fine-tuning demo because:
1. Quality is **objectively measurable** (JSON validity, field presence)
2. Base models are mediocre at it (they add explanations)
3. Improvement is dramatic and easy to visualize
4. Industrial applications are obvious to any hiring manager
