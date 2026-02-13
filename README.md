---
license: mit
base_model:
  - FacebookAI/xlm-roberta-large
language:
  - ru
tags:
  - Reasoning
  - Logical-Analysis
  - Text-Classification
  - AI-Safety
  - Evaluation
  - Judge-model
  - Argumentation
---

[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-model-blue)](https://huggingface.co/skatzR/RQA-X1.1)

# 🧠 RQA — Reasoning Quality Analyzer (R1)

**RQA** is a **judge model** designed to evaluate the *quality of reasoning in text*.  
It does **not** generate, rewrite, or explain content — instead, it **assesses whether a text contains logical problems**, and if so, **what kind**.

> **RQA is a judge, not a teacher and not a generator.**

---

## 🔍 What Problem Does RQA Solve?

Texts written by humans or LLMs can:

- sound coherent,
- use correct vocabulary,
- appear persuasive,

…but still contain **logical problems** that are:

- implicit,
- structural,
- hidden in argumentation.

**RQA focuses strictly on reasoning quality**, not on style, sentiment, or factual correctness.

---

## 🧩 Model Overview

| Property | Value |
|--------|------|
| **Model Type** | Judge / Evaluator |
| **Base Encoder** | [XLM-RoBERTa Large](https://huggingface.co/FacebookAI/xlm-roberta-large) |
| **Pooling** | Mean pooling |
| **Heads** | 2 (binary + multi-label) |
| **Language** | Russian 🇷🇺 |
| **License** | MIT |

---

## 🧠 What the Model Predicts

RQA produces **two independent signals** that are combined at inference time:

### 1️⃣ Logical Issue Detection (Binary)

- `has_issue ∈ {false, true}`
- Calibrated probability available
- Designed to answer:  
  **“Does this text contain a reasoning problem?”**

### 2️⃣ Error Type Signals (Multi-label)

The model estimates probabilities for specific error types:

- `false_causality`
- `unsupported_claim`
- `overgeneralization`
- `missing_premise`
- `contradiction`
- `circular_reasoning`

⚠️ **Important**  
Error type probabilities are **diagnostic signals**, not mandatory labels.  
They are surfaced **only if `has_issue == true`** during inference.

---

## 🟡 Hidden Logical Problems (Key Concept)

RQA explicitly distinguishes between:

### 🔴 Explicit Logical Errors
Clearly identifiable fallacies:
- invalid causal inference
- circular reasoning
- contradictions
- unsupported claims

### 🟡 Hidden Logical Problems
Texts that are:
- argumentative or persuasive,
- structurally incomplete,
- reliant on implicit assumptions,

but **do not contain a cleanly classifiable fallacy**.

Examples:
- missing or unstated premises
- rhetorical generalizations
- context-dependent claims

Hidden problems are **not misclassifications** —  
they are an **intended diagnostic category**.

---

## ⚖️ Inference Logic (Important)

The model uses **decision logic on top of raw logits**:

- Binary head decides **whether a problem exists**
- Error heads provide **type-level evidence**
- If:
  - `has_issue == false`
  - but error probabilities are non-zero  
  → the text may be flagged as **borderline** or **hidden problem**

This prevents:
- false positive error labels,
- incoherent outputs,
- over-triggering on clean factual texts.

---

## 🏗️ Architecture Details

- **Encoder**: XLM-RoBERTa Large (pretrained weights preserved)
- **Pooling**: Mean pooling (robust for long texts)
- **Two independent projections**:
  - binary reasoning head
  - multi-label error head
- Separate dropout and projections to reduce negative transfer

---

## 🎓 Training Philosophy

### 🔒 Strict Data Contract

- Logical texts **contain no errors**
- Hidden-problem texts **contain no explicit fallacies**
- Invalid samples are **removed**, not auto-corrected

### ⚖️ Balanced Difficulty

- Hidden problems ≤ **30%** of problematic texts
- Prevents collapse into vague uncertainty detection

### 🎯 Loss Design

- Binary BCE for issue detection
- Masked multi-label loss for error types
- Stability-oriented multi-task optimization

---

## 🌡️ Confidence Calibration

RQA applies **post-hoc temperature scaling**:

- Separate calibration for:
  - `has_issue`
  - each error type
- Enables:
  - meaningful probabilities
  - safe threshold tuning
  - production use without retraining

---

## 🚀 Intended Use

### ✅ Recommended for:

- Reasoning quality evaluation
- LLM output auditing
- AI safety pipelines
- Argumentation analysis
- Pre-filtering / routing systems

### ❌ Not intended for:

- Text generation
- Error correction
- Explanation or tutoring
- Grammar or style analysis
- Fact checking

---

## 🧪 Model Behavior

- Conservative by design
- Optimized for **low false positives**
- Explicitly robust to:
  - topic changes
  - writing style
  - emotional tone

RQA judges **logical structure**, not persuasion quality.

---

## 📚 Training Data (High-level)

- **Custom-built dataset**
- **Thousands of long-form argumentative texts**
- **Multiple domains and reasoning styles**
- Carefully controlled balance of:
  - logical texts
  - explicit errors
  - hidden problems

> The dataset was designed specifically for **judge behavior**, not for text generation.

---

## ⚠️ Limitations

- Logical validity ≠ factual correctness
- Purely descriptive texts may still trigger *diagnostic signals*
- Highly rhetorical or persuasive texts can be flagged as **hidden problems**
- Philosophical disagreement is **not always** a logical error

---

## 🧩 Philosophy

> **Good reasoning is not about sounding convincing —  
> it is about what actually follows from what.**

RQA is built around this principle.

---

## 🔧 Implementation Details

- Custom Hugging Face architecture (`modeling_rqa.py`)
- Requires:
  - `trust_remote_code=True`
- Uses `safetensors`
- No `.bin` weights (this is expected behavior)

---

## 🚀 Quick Start

```python
import torch
from transformers import AutoTokenizer, AutoModel

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(
    "skatzR/RQA-R1",
    trust_remote_code=True
)

model = AutoModel.from_pretrained(
    "skatzR/RQA-R1",
    trust_remote_code=True
).to(device)

model.eval()

```
---

## 🧠 Reference Inference Logic

RQA is designed to be used with **explicit post-processing logic**, including:

- temperature scaling
- thresholding
- disagreement diagnostics
- hidden-problem detection

A **fully working reference implementation** is provided here:


👉 **[📄 inference.py](https://huggingface.co/skatzR/RQA-X1.1/blob/main/inference.py) — Reference Inference Implementation**

👉 **[🤖Try it live ](https://reasoning-quality-analyzer.streamlit.app/) — Instant logical error detection in your browser. Powered by Streamlit Cloud.**



---
## ✅ Example
```
📄 Текст:
После того как в городе открыли новый торговый центр, увеличилось количество разводов. 
Следовательно, открытие торгового центра разрушает семьи.

🔎 Обнаружена проблема: ДА (100.00%)

❌ Явные логические ошибки:
  • Ложная причинно-следственная связь — 95.95%

📊 Disagreement: 0.034
```
---

## 📜 License

MIT

---
