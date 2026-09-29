# MedQA — 3-Stage CRAG Pipeline

A progressive, 3-stage Retrieval-Augmented Generation (RAG) system evaluated on the **MedQA USMLE 4-option** benchmark dataset. Each stage builds on the previous one — starting from a plain RAG baseline all the way to a quantized CRAG system that runs on a local consumer GPU.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Dataset](#dataset)
4. [Shared Utilities — medqa_shared.py](#shared-utilities--medqa_sharedpy)
5. [Stage 1 — Plain RAG Baseline (No Evaluator)](#stage-1--plain-rag-baseline-no-evaluator)
6. [Stage 2 — CRAG (Train DistilBERT + Insert into RAG)](#stage-2--crag-train-distilbert--insert-into-rag)
7. [Stage 3 — Quantized CRAG (4-bit DistilBERT, Local PC)](#stage-3--quantized-crag-4-bit-distilbert-local-pc)
8. [Results Summary](#results-summary)
9. [How to Run](#how-to-run)
10. [Requirements](#requirements)

---

## Project Overview

The goal of this project is to systematically improve the accuracy of a medical question-answering system on the USMLE MedQA benchmark through three progressively more powerful stages:

| Stage | Description | Where Run | Accuracy |
|-------|-------------|-----------|----------|
| **Stage 1** | Plain RAG — BM25 + FAISS + Flan-T5, **no evaluator** | Local / Colab | **27.02%** |
| **Stage 2** | CRAG — Train DistilBERT evaluator, then plug into RAG | Google Colab (Tesla T4) | **65.04%** |
| **Stage 3** | Quantized CRAG — Compress DistilBERT to 4-bit, run locally | Local PC (GTX 1650) | **62.92%** |

The key insight is that adding a **trained relevance evaluator (DistilBERT)** that filters out bad retrieved snippets before passing them to the generator dramatically improves accuracy — from 27% to 65%. Compressing the evaluator to 4-bit costs only ~2% accuracy but makes the system runnable on a laptop GPU.

---

## Repository Structure

```
medqa/
│
├── medqa_shared.py            # Shared utilities (retriever, embedder, corpus builder, prompt, parser)
├── stage1.py                  # Stage 1: Plain RAG evaluation (no evaluator)
├── stage2.py                  # Stage 2: Train DistilBERT evaluator + full CRAG evaluation
├── stage3.py                  # Stage 3: 4-bit quantized CRAG evaluation (local PC)
│
├── stage1_detailed_report.txt # Human-readable Stage 1 results report
├── stage2_detailed_report.txt # Human-readable Stage 2 results report
├── stage3_detailed_report.txt # Human-readable Stage 3 results report
│
├── stage1_predictions.json    # Detailed per-question predictions from Stage 1
│
├── my_medical_evaluator/      # Saved FP16 DistilBERT checkpoint (output of stage2.py)
│
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

> **Note:** `stage2_predictions.json` and `stage3_predictions.json` are generated at runtime (large files).

---

## Dataset

**Name:** `GBaker/MedQA-USMLE-4-options` (Hugging Face)

This is a 4-option multiple-choice medical QA dataset based on USMLE Step exams.

| Split | Size | Used For |
|-------|------|----------|
| `train` | 10,178 questions | Knowledge corpus (first 10,000) + DistilBERT training pairs |
| `test` | 1,273 questions | Evaluation in all 3 stages |

> **Important:** The test split is **never** added to the retrieval corpus. The corpus is always built exclusively from the **training split** to prevent data leakage.

Each example has the format:
```json
{
  "question": "A 67-year-old man with a 40-pack-year...",
  "options": { "A": "Repeat CT", "B": "Biopsy", "C": "PET-CT", "D": "VATS" },
  "answer": "PET-CT",
  "answer_idx": "C"
}
```

---

## Shared Utilities — `medqa_shared.py`

All three stages import from this single file. It contains everything shared across the pipeline so that results are perfectly reproducible and consistent between stages.

### `CORPUS_SIZE = 10,000`
The fixed number of training passages used as the knowledge base for retrieval in all stages.

---

### `build_corpus(train_dataset)`
Builds the 10,000-document knowledge corpus from the training split.

Each document is formatted as:
```
Clinical Scenario: {question}
Medical Finding & Answer: {answer}
```

- Always takes the **first 10,000** training examples.
- **Never** uses the test split.

---

### `DenseEmbedder`
Uses `sentence-transformers/all-MiniLM-L6-v2` to create dense vector embeddings.

- **Mean pooling** over all token embeddings.
- **L2-normalised** for cosine similarity via inner product in FAISS.
- Batch size: 64 for corpus encoding, 1 for query encoding at inference.

---

### `HybridRetriever`
Combines **BM25** (keyword search) and **FAISS** (semantic search) using **Reciprocal Rank Fusion (RRF, k=60)**.

**How RRF works:**
```
RRF_score(doc) = 1/(k + rank_BM25) + 1/(k + rank_FAISS)
```
- Each retriever independently ranks all documents.
- Scores are summed — documents appearing high in both lists get the highest final score.
- Returns top-K documents by combined RRF score.

**Caching:** Indices are saved to `.cache_index/` on the first run and reloaded automatically on subsequent runs. Stale caches (wrong corpus size) are auto-detected and rebuilt.

---

### `format_prompt(question, options, snippets)`
Formats the input for `google/flan-t5-base`:
```
Context:
Snippet 1: ...
Snippet 2: ...
Snippet 3: ...

Question: A 67-year-old man...

Options:
(A) ...
(B) ...
(C) ...
(D) ...

Based on the context, select the correct option letter (A, B, C, or D).
Answer:
```

---

### `parse_prediction(generated_text, options)`
Robustly extracts the answer letter (A/B/C/D) from raw model output using 3 strategies in order:
1. **Direct regex:** Matches `A`, `(A)`, `A.`, `A)` patterns.
2. **Fuzzy match:** Compares output against option text values.
3. **First A-D character:** Scans the output for the first valid letter.
4. **Fallback:** Returns `"A"` if nothing matches.

---

## Stage 1 — Plain RAG Baseline (No Evaluator)

**File:** `stage1.py`

### What it does
The simplest possible RAG pipeline. No evaluator. No filtering. Every retrieved snippet goes directly to the generator.

### Pipeline Flow

```
Test Question
     |
     v
HybridRetriever (BM25 + FAISS, RRF k=60)
     |  retrieve top-3 snippets from 10,000-doc corpus
     v
format_prompt(question, options, top-3 snippets)
     |
     v
google/flan-t5-base (generator)
     |  max_new_tokens=16, greedy decoding
     v
parse_prediction --> Answer Letter (A/B/C/D)
     |
     v
Compare with ground truth --> Correct/Incorrect
```

### Key Configuration

| Parameter | Value |
|-----------|-------|
| Generator | `google/flan-t5-base` |
| Embedding Model | `sentence-transformers/all-MiniLM-L6-v2` |
| Retrieval | BM25 + FAISS (RRF k=60) |
| Top-K Snippets | 3 |
| Corpus Size | 10,000 training passages |
| Evaluator | **None** |
| Batch Size (inference) | 8 |

### What is NOT in Stage 1
- No DistilBERT
- No relevance scoring
- No filtering of retrieved snippets
- All top-3 retrieved snippets are passed to the generator regardless of quality

### Outputs
| File | Description |
|------|-------------|
| `stage1_detailed_report.txt` | Human-readable report with accuracy + 5 sample breakdowns |
| `stage1_predictions.json` | All 1,273 predictions with retrieved snippets |

### Results

| Metric | Value |
|--------|-------|
| Total Questions | 1,273 |
| Correct Predictions | 344 |
| **Final Accuracy** | **27.02%** |
| Time Elapsed | 3,841.27s (~3.02s per question) |

### Why is 27% so low?
The retriever often pulls snippets that match keywords in the question but are not the correct clinical context. Without any filtering, the generator is misled by irrelevant or conflicting information. For example, a question about a lung nodule retrieves snippets about surgical resection, leading flan-t5 to answer "D" (surgery) instead of the correct "C" (PET-CT staging first). The retriever is working, but without quality filtering its output hurts more than it helps.

---

## Stage 2 — CRAG (Train DistilBERT + Insert into RAG)

**File:** `stage2.py`
**Run on:** Google Colab (Tesla T4 GPU)

Stage 2 has two parts running in sequence within the same script:
- **Part A:** Train the DistilBERT relevance evaluator
- **Part B:** Use the trained evaluator inside the RAG loop (making it CRAG) to evaluate all 1,273 test questions

---

### Part A — Training the DistilBERT Evaluator

#### What is the evaluator?
A binary classifier (`distilbert-base-uncased`) that scores whether a retrieved document snippet is **relevant** to a given medical question. Score >= 0.50 = keep the snippet. Score < 0.50 = discard (purge) it.

#### Training Data Construction
From the 10,178 training questions, two types of pairs are created per question:

**Positive pair (label = 1 — relevant):**
```
Question: "A 25-year-old woman presents with episodic headaches..."
Document: "Medical question: A 25-year-old woman...
           Relevant medical answer: Phenoxybenzamine"
```

**Negative pair (label = 0 — not relevant):**
```
Question: "A 25-year-old woman presents with episodic headaches..."
Document: "Medical question: A 25-year-old woman...
           Potential medical information: Propranolol"   <- a wrong answer option
```

This produces **20,356 pairs** (10,178 positive + 10,178 negative), randomly shuffled with seed 42.

#### Train/Validation Split
- 90% training: ~18,320 pairs
- 10% validation: ~2,036 pairs

#### Model Architecture

| Parameter | Value |
|-----------|-------|
| Base Model | `distilbert-base-uncased` |
| Task | Binary sequence classification (relevant / not-relevant) |
| Output | 2 logits -> softmax -> probability of relevance |
| Max Sequence Length | 256 tokens |

#### Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Epochs | 3 |
| Batch Size | 32 |
| Learning Rate | 2e-5 |
| Optimizer | AdamW (weight_decay=0.01) |
| Scheduler | Linear warmup (10% of total steps) |
| Mixed Precision | FP16 (AMP) |
| Gradient Clipping | 1.0 |
| Seed | 42 |
| Compute | Tesla T4 (Google Colab) |

#### Training Results

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|-----------|-----------|----------|---------|
| 1/3 | 0.5209 | 63.26% | 0.4891 | 63.95% |
| 2/3 | 0.4854 | 65.36% | 0.4887 | 65.32% |
| 3/3 | 0.4853 | 64.65% | 0.4888 | 65.32% |

**Final Validation Accuracy: 65.32%**
Training Time: 476.00s (~8 minutes on Tesla T4)

#### Model Saving
The trained model is saved as an **FP16 checkpoint** to `./my_medical_evaluator/` using:
```python
model.half()             # Convert to FP16
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
```
FP16 is used to halve storage and memory usage while preserving near-identical accuracy vs FP32.

---

### Part B — Full CRAG Evaluation

#### What is CRAG?
**Corrective RAG (CRAG)** adds a learned evaluator between the retriever and generator. Instead of blindly feeding all retrieved snippets to the generator, CRAG scores each snippet and only passes the relevant ones through.

#### CRAG Pipeline Flow

```
Test Question
     |
     v
HybridRetriever (BM25 + FAISS, RRF k=60)
     |  retrieve top-5 candidate snippets from 10,000-doc corpus
     v
For each candidate snippet:
  +---------------------------------------------------+
  | DistilBERT Evaluator (FP16)                       |
  | Input:  [question] + [snippet]                    |
  | Output: relevance probability (0.0 to 1.0)        |
  | If prob >= 0.50 --> KEEP (retained)               |
  | If prob <  0.50 --> DISCARD (purged)              |
  +---------------------------------------------------+
     |
     v
If zero snippets retained --> fallback: keep top-2 by retrieval rank
     |
     v
format_prompt(question, options, retained_snippets)
     |
     v
google/flan-t5-base (generator)
     |  max_new_tokens=16, greedy decoding
     v
parse_prediction --> Answer Letter (A/B/C/D)
     |
     v
Compare with ground truth --> Correct/Incorrect
```

#### Key Configuration

| Parameter | Value |
|-----------|-------|
| Retrieval Corpus | First 10,000 training passages |
| Retriever | BM25 + FAISS (RRF k=60) |
| Top-K Candidates | 5 |
| Relevance Threshold | >= 0.50 |
| Fallback Policy | Top-2 by retrieval rank if none pass |
| Generator | `google/flan-t5-base` |
| Evaluator | FP16 DistilBERT (`./my_medical_evaluator/`) |

#### Outputs
| File | Description |
|------|-------------|
| `stage2_detailed_report.txt` | Training table + CRAG evaluation results |
| `stage2_predictions.json` | All predictions with retained snippets + relevance scores |

#### Results

| Metric | Value |
|--------|-------|
| Total Questions | 1,273 |
| Correct Predictions | 828 |
| **CRAG Accuracy** | **65.04%** |
| Total Time (CRAG) | 7,638.42s (~6.00s per question) |
| Total Candidates Evaluated | 6,365 |
| Candidates Purged | 2,281 (35.8%) |
| Candidates Retained | 4,084 (64.2%) |

#### Why the massive improvement (27% -> 65%)?
The evaluator successfully removes ~36% of retrieved snippets that are off-topic or misleading. The generator now only sees high-quality, relevant context, making it far more likely to pick the correct answer. The 35.8% purge rate shows the retriever often returns plausible-but-wrong snippets — the evaluator catches these.

---

## Stage 3 — Quantized CRAG (4-bit DistilBERT, Local PC)

**File:** `stage3.py`
**Run on:** Local PC — NVIDIA GeForce GTX 1650

Stage 3 takes the trained FP16 DistilBERT evaluator from Stage 2 and **compresses it to 4-bit** so it can run efficiently on a consumer GPU with limited VRAM.

### Why Quantize?
The GTX 1650 has only 4GB VRAM. Quantizing the evaluator to 4-bit frees up memory for the rest of the pipeline (embedder + flan-t5) and speeds up inference.

### Quantization Method: NF4 (4-bit via bitsandbytes)

**NF4 (NormalFloat4)** is a data type designed specifically for quantizing neural networks. It maps each weight to one of 16 values optimally distributed for normally-distributed (neural network) weights.

```python
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

eval_model = AutoModelForSequenceClassification.from_pretrained(
    "./my_medical_evaluator",
    quantization_config=quantization_config,
    device_map="auto",
)
```

- Weights are stored in 4-bit but **dequantized to FP16** at compute time.
- `device_map="auto"` automatically places the model on GPU.

### CPU Fallback (int8)
If no CUDA GPU is available, the script automatically falls back to `torch.quantization.quantize_dynamic` (int8 dynamic quantization on CPU). This fallback exists for portability but was **not** used for the reported numbers — those used the NF4 GPU path on the GTX 1650.

### Compression Results

| | Value |
|-|-------|
| Before quantization (FP16) | 261.8 MB |
| After quantization (4-bit NF4) | 35.2 MB |
| **Compression ratio** | **7.4x smaller** |

### CRAG Loop (Identical to Stage 2)
The evaluation loop is exactly the same as Stage 2:
- Top-5 candidate retrieval
- Score each with the **4-bit quantized** evaluator (threshold >= 0.50)
- Fallback to top-2 if none pass
- Generate with flan-t5-base
- Parse answer letter

### Key Configuration

| Parameter | Value |
|-----------|-------|
| Evaluator Checkpoint | `./my_medical_evaluator/` (from Stage 2) |
| Quantization | bitsandbytes NF4 (4-bit) |
| Generator | `google/flan-t5-base` |
| Top-K Candidates | 5 |
| Relevance Threshold | >= 0.50 |
| Fallback | Top-2 by retrieval rank |
| Device | NVIDIA GeForce GTX 1650 |

### Outputs
| File | Description |
|------|-------------|
| `stage3_detailed_report.txt` | Quantization details + CRAG evaluation results |
| `stage3_predictions.json` | All predictions with quantization method recorded |

### Results

| Metric | Value |
|--------|-------|
| Total Questions | 1,273 |
| Correct Predictions | 801 |
| **Final CRAG Accuracy** | **62.92%** |
| Total Time | 3,961.44s (~3.11s per question) |
| Total Candidates Evaluated | 6,365 |
| Candidates Purged | 2,447 (38.4%) |
| Candidates Retained | 3,918 (61.6%) |

### Why only ~2% accuracy drop from Stage 2?
NF4 quantization preserves the relative ordering of weights very well because it uses an information-theoretically optimal distribution of quantization levels for neural network weights. The evaluator's binary classification task (relevant vs. not) is relatively robust — it does not need extreme numerical precision to make a correct 0/1 decision.

---

## Results Summary

### Accuracy Progression

```
Stage 1  RAG, no evaluator         27.02%   [=====                   ]
Stage 2  CRAG, FP16 DistilBERT     65.04%   [=============           ]
Stage 3  CRAG, 4-bit DistilBERT    62.92%   [============            ]
```

### Side-by-Side Comparison

| Metric | Stage 1 | Stage 2 | Stage 3 |
|--------|---------|---------|---------|
| Evaluator | None | FP16 DistilBERT | 4-bit NF4 DistilBERT |
| Evaluator Size | N/A | 261.8 MB | 35.2 MB |
| Top-K Candidates | 3 | 5 | 5 |
| Relevance Filtering | No | Yes | Yes |
| Snippets Purged | 0% | 35.8% | 38.4% |
| **Accuracy** | **27.02%** | **65.04%** | **62.92%** |
| Time per Question | 3.02s | 6.00s | 3.11s |
| Run Location | Any | Colab T4 | GTX 1650 (local) |

### Key Takeaways

1. **The evaluator is the game-changer.** Adding DistilBERT to filter retrieved snippets delivers a **+38 percentage point** accuracy gain.
2. **4-bit quantization is nearly free.** Stage 3 sacrifices only **~2%** accuracy while achieving a **7.4x model size reduction**, making the system viable on consumer hardware.
3. **Stage 3 is actually faster than Stage 2** (~3.1s vs ~6.0s per question) because the 4-bit model runs inference more efficiently on the GTX 1650.
4. **No data leakage** — the test split was never used to build the retrieval corpus in any stage.

---

## How to Run

### Prerequisites
- Python 3.9+
- NVIDIA GPU with CUDA (required for Stage 2 and the NF4 path in Stage 3)
- For Stage 2: Google Colab recommended (Tesla T4)
- For Stage 3: Any NVIDIA GPU (tested on GTX 1650)

### Step 1 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Run Stage 1 (Plain RAG Baseline)
```bash
python stage1.py
```
Outputs: `stage1_detailed_report.txt`, `stage1_predictions.json`

### Step 3 — Run Stage 2 (Train Evaluator + CRAG) on Colab
Upload `stage2.py` and `medqa_shared.py` to Google Colab, then run:
```bash
python stage2.py
```
Outputs: `stage2_detailed_report.txt`, `stage2_predictions.json`, `./my_medical_evaluator/`

Download `./my_medical_evaluator/` to your local machine after it finishes.

### Step 4 — Run Stage 3 (Quantized CRAG) on Local PC
Make sure `./my_medical_evaluator/` is in the same directory as `stage3.py`, then run:
```bash
python stage3.py
```
Outputs: `stage3_detailed_report.txt`, `stage3_predictions.json`

---

## Requirements

See `requirements.txt`. Key packages:

| Package | Purpose |
|---------|---------|
| `torch` | Deep learning framework |
| `transformers` | Flan-T5, DistilBERT, BitsAndBytes quantization |
| `bitsandbytes` | 4-bit NF4 quantization (Stage 3 GPU path) |
| `datasets` | Hugging Face MedQA dataset loading |
| `faiss-cpu` / `faiss-gpu` | Dense vector index for semantic search |
| `rank-bm25` | BM25 sparse keyword retrieval |
| `sentence-transformers` | all-MiniLM-L6-v2 dense embeddings |
| `numpy` | Numerical operations |
| `tqdm` | Progress bars |

---

*This project demonstrates that a lightweight trained relevance evaluator (DistilBERT) can dramatically improve RAG accuracy on medical QA tasks, and that 4-bit quantization makes such evaluators deployable on consumer-grade hardware with minimal accuracy loss.*
