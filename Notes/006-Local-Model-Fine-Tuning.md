# 🧠 Supervised Fine-Tuning (SFT) on Local/Open-Source Models

## 📌 Overview

**Supervised Fine-Tuning (SFT)** is the process of adapting a pretrained model (e.g., LLaMA, Mistral, Falcon) to perform better on a specific task by training it further on labeled data. Unlike pretraining, SFT is task-specific and usually requires less compute.

---

## 🧩 Why Fine-Tune?

* **Adaptation** to a specific domain (e.g., medical, legal).
* **Improve accuracy** on downstream tasks (e.g., Q\&A, summarization).
* **Teach custom behavior** using your own dataset.
* **Control model outputs** more precisely (e.g., formatting, tone).

---

## 🏗️ General Pipeline

### 1. **Select a Base Model**

* Examples:

  * `Meta-LLaMA`, `Mistral`, `Falcon`, `Gemma`
  * Hosted via Hugging Face (`transformers`) or `llama.cpp` for local inference.

### 2. **Prepare Dataset**

* Must be **labeled** and task-aligned.
* Format: JSON, JSONL, CSV, etc.
* Structure example (Instruction Tuning):

```json
{
  "instruction": "Summarize this paragraph...",
  "input": "Long paragraph here.",
  "output": "Short summary here."
}
```

* Tools:

  * `datasets` library (Hugging Face)
  * `jsonlines`, `pandas`, custom preprocessing scripts

### 3. **Tokenize Data**

* Use tokenizer matching the base model:

  ```python
  tokenizer = AutoTokenizer.from_pretrained("model-name")
  tokenized = tokenizer(text, truncation=True, padding="max_length")
  ```

### 4. **Load Model**

* Transformers example:

  ```python
  from transformers import AutoModelForCausalLM

  model = AutoModelForCausalLM.from_pretrained("model-name")
  ```

### 5. **Apply LoRA (optional)**

* Reduces GPU memory & training time
* Libraries: `peft`, `QLoRA`, `bitsandbytes`

  ```python
  from peft import get_peft_model, LoraConfig
  peft_model = get_peft_model(model, LoraConfig(...))
  ```

### 6. **Train**

* Use `Trainer`, `SFTTrainer`, or `accelerate`:

  ```python
  trainer = Trainer(model=model, ...)
  trainer.train()
  ```
* Log metrics (loss, perplexity)
* Save checkpoints and final model

---

## 🛠️ Tools & Libraries

| Tool            | Purpose                                |
| --------------- | -------------------------------------- |
| 🤗 Transformers | Load models, tokenizers                |
| 🤗 Datasets     | Manage datasets                        |
| 🦙 PEFT         | Parameter-efficient fine-tuning (LoRA) |
| bitsandbytes    | 8-bit/4-bit quantization               |
| 🤗 Accelerate   | Multi-GPU training                     |
| WandB           | Experiment tracking (optional)         |

---

## ⚙️ Common Configs

* **Batch Size**: 2–32 (depends on GPU)
* **Learning Rate**: `2e-5` to `1e-4`
* **Epochs**: 3–10
* **Max Length**: 512–2048 tokens
* **Gradient Accumulation**: To simulate larger batches
* **Save Steps**: Save checkpoints during training

---

## 🧪 Evaluation

* Use held-out validation set.
* Metrics: BLEU, ROUGE, accuracy, perplexity
* Optionally, test on real-world prompts

---

## 💾 Saving & Using Fine-Tuned Model

* Save with `trainer.save_model("path")`
* Push to Hugging Face Hub (optional)
* Inference:

  ```python
  model.eval()
  input_ids = tokenizer(prompt, return_tensors="pt").input_ids
  outputs = model.generate(input_ids)
  ```

---

## ⚠️ Gotchas & Tips

* Always match tokenizer with model!
* Quantized models (4-bit, 8-bit) may not support full backprop.
* Avoid overfitting: use early stopping or monitor loss.
* Monitor GPU VRAM and CPU RAM usage.

---
