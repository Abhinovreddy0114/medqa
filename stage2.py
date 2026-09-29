"""
stage2.py

Stage 2: CRAG DistilBERT Relevance Evaluator — Training + Full CRAG Evaluation.

Part A — Training (unchanged hyperparameters):
  - Dataset: GBaker/MedQA-USMLE-4-options (full 10,178 training questions)
  - 20,356 positive/negative pairs (correct answer = relevant, one random wrong option = irrelevant)
  - Model: distilbert-base-uncased, 3 epochs, batch_size=32, lr=2e-5, max_len=256
  - AdamW with weight_decay=0.01, linear warmup, FP16 mixed precision (on GPU)
  - 90/10 train/validation split
  - Saved FP16 model -> ./my_medical_evaluator

Part B — Full end-to-end CRAG Evaluation Loop (NEW):
  - Retrieval corpus: first 10,000 training passages (NEVER test split)
  - HybridRetriever (BM25 + FAISS, RRF k=60) from medqa_shared
  - Top-5 candidate retrieval per question
  - DistilBERT evaluator with FP16 autocast scores each candidate
  - Keep candidates with relevance probability >= 0.50; fallback to top-2 if none pass
  - Generate answer with google/flan-t5-base using shared format_prompt
  - Parse answer letter with shared parse_prediction
  - Evaluate on all 1,273 test questions

Outputs:
  - stage2_detailed_report.txt  (training table + CRAG eval section)
  - stage2_predictions.json     (same schema as stage1, + retained_snippets, relevance_scores)
"""

import json
import logging
import os
import random
import time
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from medqa_shared import (
    CORPUS_SIZE,
    DenseEmbedder,
    HybridRetriever,
    build_corpus,
    format_prompt,
    parse_prediction,
)

# ============================================================
# CONFIGURATION (do not change)
# ============================================================

SEED = 42
MODEL_NAME = "distilbert-base-uncased"
BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 2e-5
MAX_LENGTH = 256
OUTPUT_DIR = "./my_medical_evaluator"
RELEVANCE_THRESHOLD = 0.50
TOP_CANDIDATES = 5
FALLBACK_K = 2
INDEX_DIR = ".cache_index"
GENERATOR_MODEL = "google/flan-t5-base"
DATASET_NAME = "GBaker/MedQA-USMLE-4-options"

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("MedQA-Stage2")

# ============================================================
# SEED
# ============================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {device}")
if device.type == "cuda":
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================================
# DATASET — Training pairs
# ============================================================

logger.info("Loading MedQA dataset...")
dataset = load_dataset(DATASET_NAME, split="train")
logger.info(f"Training questions available: {len(dataset):,}")


def create_pairs(dataset, seed: int = 42) -> List[Dict]:
    rng = random.Random(seed)
    pairs = []

    for item in dataset:
        question = item["question"]
        correct_answer = item["answer"]
        options = item["options"]

        # ---------- POSITIVE ----------
        positive_document = (
            f"Medical question: {question}\n"
            f"Relevant medical answer: {correct_answer}"
        )
        pairs.append({"question": question, "document": positive_document, "label": 1})

        # ---------- NEGATIVE ----------
        wrong_answers = [
            str(opt).strip()
            for key, opt in options.items()
            if str(opt).strip() != str(correct_answer).strip()
        ]
        if wrong_answers:
            wrong_answer = rng.choice(wrong_answers)
            negative_document = (
                f"Medical question: {question}\n"
                f"Potential medical information: {wrong_answer}"
            )
            pairs.append({"question": question, "document": negative_document, "label": 0})

    rng.shuffle(pairs)
    return pairs


logger.info("Creating relevance pairs...")
pairs = create_pairs(dataset)
logger.info(f"Total pairs: {len(pairs):,}")
positive = sum(1 for x in pairs if x["label"] == 1)
negative = sum(1 for x in pairs if x["label"] == 0)
logger.info(f"Positive: {positive:,} | Negative: {negative:,}")

# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

random.Random(SEED).shuffle(pairs)
split_index = int(len(pairs) * 0.90)
train_pairs = pairs[:split_index]
val_pairs = pairs[split_index:]
logger.info(f"Training pairs: {len(train_pairs):,}")
logger.info(f"Validation pairs: {len(val_pairs):,}")

# ============================================================
# TOKENIZER
# ============================================================

logger.info(f"Loading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ============================================================
# DATASET CLASS
# ============================================================

class RelevanceDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_length: int = 256):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        item = self.pairs[idx]
        encoded = self.tokenizer(
            item["question"],
            item["document"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(item["label"], dtype=torch.long),
        }


train_dataset_obj = RelevanceDataset(train_pairs, tokenizer, MAX_LENGTH)
val_dataset_obj = RelevanceDataset(val_pairs, tokenizer, MAX_LENGTH)

# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset_obj, batch_size=BATCH_SIZE, shuffle=True, drop_last=False
)
val_loader = DataLoader(val_dataset_obj, batch_size=BATCH_SIZE, shuffle=False)

# ============================================================
# MODEL
# ============================================================

logger.info(f"Loading model: {MODEL_NAME}")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.to(device)

# ============================================================
# OPTIMIZER & SCHEDULER
# ============================================================

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(total_steps * 0.1),
    num_training_steps=total_steps,
)

# ============================================================
# FP16 MIXED PRECISION
# ============================================================

use_amp = device.type == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

# ============================================================
# VALIDATION HELPER
# ============================================================

def evaluate(model, loader):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_examples = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            if use_amp:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = criterion(outputs.logits, labels)
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)

            predictions = torch.argmax(outputs.logits, dim=1)
            total_correct += (predictions == labels).sum().item()
            total_examples += labels.size(0)
            total_loss += loss.item() * labels.size(0)

    accuracy = (total_correct / total_examples) * 100
    average_loss = total_loss / total_examples
    return average_loss, accuracy

# ============================================================
# TRAINING
# ============================================================

logger.info("=" * 70)
logger.info("STARTING STAGE 2 DISTILBERT TRAINING")
logger.info("=" * 70)

training_start = time.time()
train_losses, val_losses, train_accs, val_accs = [], [], [], []

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0
    correct = 0
    total = 0
    logger.info(f"Epoch {epoch + 1}/{EPOCHS}")

    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        if use_amp:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = nn.CrossEntropyLoss()(outputs.logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = nn.CrossEntropyLoss()(outputs.logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()
        predictions = torch.argmax(outputs.logits, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        running_loss += loss.item() * labels.size(0)

    train_loss = running_loss / total
    train_acc = (correct / total) * 100
    val_loss, val_acc = evaluate(model, val_loader)

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    logger.info(
        f"Epoch {epoch + 1}: "
        f"Train Loss={train_loss:.4f} | Train Acc={train_acc:.2f}% | "
        f"Val Loss={val_loss:.4f} | Val Acc={val_acc:.2f}%"
    )

training_elapsed = time.time() - training_start

# ============================================================
# SAVE FP16 MODEL
# ============================================================

logger.info("Saving FP16 evaluator...")
os.makedirs(OUTPUT_DIR, exist_ok=True)
model.half()  # Convert to FP16 for storage
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
logger.info(f"FP16 model saved to '{OUTPUT_DIR}'.")

# ============================================================
# PART B: FULL CRAG EVALUATION LOOP
# ============================================================

logger.info("=" * 70)
logger.info("STARTING STAGE 2 CRAG EVALUATION LOOP")
logger.info("=" * 70)

# Load test split (used only for questions, never added to corpus)
logger.info("Loading MedQA test split (1,273 questions)...")
test_dataset = load_dataset(DATASET_NAME, split="test")
total_test = len(test_dataset)
logger.info(f"Test questions: {total_test:,}")

# Build 10,000-doc corpus from training split
logger.info("Loading training split for corpus construction...")
train_raw = load_dataset(DATASET_NAME, split="train")
documents = build_corpus(train_raw)

# Build hybrid retriever
embedder = DenseEmbedder(device=str(device))
retriever = HybridRetriever(
    documents=documents,
    embedder=embedder,
    index_dir=INDEX_DIR,
    k_rrf=60,
    corpus_size=CORPUS_SIZE,
)

# Load generator
logger.info(f"Loading generator model '{GENERATOR_MODEL}'...")
gen_tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL)
gen_model = AutoModelForSeq2SeqLM.from_pretrained(GENERATOR_MODEL).to(device)
gen_model.eval()

# Reload evaluator from disk in FP16
logger.info(f"Reloading FP16 evaluator from '{OUTPUT_DIR}' for inference...")
eval_model = AutoModelForSequenceClassification.from_pretrained(
    OUTPUT_DIR, torch_dtype=torch.float16
).to(device)
eval_model.eval()
eval_tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)

# CRAG evaluation
logger.info(f"Running CRAG evaluation on {total_test:,} test questions...")
crag_correct = 0
total_candidates_eval = 0
total_purged = 0
crag_results = []
crag_start = time.time()

for item in tqdm(test_dataset, desc="Stage2 CRAG Eval", unit="q"):
    question = item["question"]
    options = item["options"]
    ground_truth = item.get("answer_idx") or item.get("answer")

    # --- Retrieve top-5 candidates ---
    candidates = retriever.retrieve(question, top_k=TOP_CANDIDATES)

    # --- Score each candidate with FP16 DistilBERT evaluator ---
    retained_snippets = []
    relevance_scores = []

    for cand in candidates:
        total_candidates_eval += 1
        enc = eval_tokenizer(
            question, cand,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            if use_amp:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    outputs = eval_model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                outputs = eval_model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits.float(), dim=-1)
            relevance_score = probs[0][1].item()

        relevance_scores.append(relevance_score)
        if relevance_score >= RELEVANCE_THRESHOLD:
            retained_snippets.append(cand)
        else:
            total_purged += 1

    # --- Fallback to top-2 by retrieval rank if nothing passes ---
    if not retained_snippets:
        retained_snippets = candidates[:FALLBACK_K]

    # --- Generate with flan-t5-base ---
    prompt = format_prompt(question, options, retained_snippets)
    gen_inputs = gen_tokenizer(
        prompt, return_tensors="pt", max_length=1024, truncation=True
    ).to(device)
    with torch.no_grad():
        gen_outputs = gen_model.generate(
            **gen_inputs, max_new_tokens=16, num_beams=1, do_sample=False
        )
    raw_gen = gen_tokenizer.decode(gen_outputs[0], skip_special_tokens=True)
    predicted_letter = parse_prediction(raw_gen, options)

    # --- Resolve ground truth letter ---
    target_letter = str(ground_truth).strip().upper()
    if target_letter not in ("A", "B", "C", "D"):
        for opt_k, opt_v in options.items():
            if opt_v.strip() == str(ground_truth).strip():
                target_letter = opt_k
                break

    is_correct = predicted_letter == target_letter
    if is_correct:
        crag_correct += 1

    crag_results.append({
        "question": question,
        "options": options,
        "target": target_letter,
        "prediction": predicted_letter,
        "raw_generation": raw_gen,
        "is_correct": is_correct,
        "retrieved_snippets": candidates,
        "retained_snippets": retained_snippets,
        "relevance_scores": relevance_scores,
    })

crag_elapsed = time.time() - crag_start
crag_accuracy = (crag_correct / total_test) * 100.0

logger.info(f"Stage 2 CRAG Accuracy: {crag_accuracy:.2f}% ({crag_correct}/{total_test})")

# ============================================================
# SAVE PREDICTIONS JSON
# ============================================================

predictions_summary = {
    "dataset_name": DATASET_NAME,
    "test_split": "test",
    "total_questions": total_test,
    "correct_predictions": crag_correct,
    "crag_accuracy_percentage": round(crag_accuracy, 2),
    "time_elapsed_seconds_crag": round(crag_elapsed, 2),
    "generator_model": GENERATOR_MODEL,
    "evaluator_model": MODEL_NAME,
    "relevance_threshold": RELEVANCE_THRESHOLD,
    "top_candidates": TOP_CANDIDATES,
    "fallback_k": FALLBACK_K,
    "total_candidates_evaluated": total_candidates_eval,
    "total_candidates_purged": total_purged,
    "detailed_results": crag_results,
}

with open("stage2_predictions.json", "w", encoding="utf-8") as f:
    json.dump(predictions_summary, f, indent=2, ensure_ascii=False)
logger.info("stage2_predictions.json saved.")

# ============================================================
# REPORT
# ============================================================

gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
purge_pct = (total_purged / max(total_candidates_eval, 1)) * 100.0

report = f"""================================================================================
                    MEDQA STAGE 2: CRAG EVALUATOR
================================================================================
Dataset:                  {DATASET_NAME}
Training Questions:       {len(dataset):,}
Training Pairs:           {len(pairs):,}
Positive Pairs:           {positive:,}
Negative Pairs:           {negative:,}

Model Architecture:       DistilBERT (distilbert-base-uncased)
Training Epochs:          {EPOCHS}
Batch Size:               {BATCH_SIZE}
Learning Rate:            {LEARNING_RATE}
Maximum Sequence Length:  {MAX_LENGTH}

Compute Accelerator:      {gpu_name}
Mixed Precision:          FP16

--------------------------------------------------------------------------------
TRAINING PERFORMANCE
--------------------------------------------------------------------------------
"""

for i in range(EPOCHS):
    report += (
        f"Epoch {i + 1}/{EPOCHS} -> "
        f"Train Loss: {train_losses[i]:.4f} | "
        f"Train Acc: {train_accs[i]:.2f}% | "
        f"Val Loss: {val_losses[i]:.4f} | "
        f"Val Acc: {val_accs[i]:.2f}%\n"
    )

report += f"""
--------------------------------------------------------------------------------
FINAL VALIDATION ACCURACY: {val_accs[-1]:.2f}%

Training Time:             {training_elapsed:.2f}s
Saved Model:               {OUTPUT_DIR}/
--------------------------------------------------------------------------------
STATUS: Stage 2 evaluator training completed.
================================================================================

================================================================================
                    MEDQA STAGE 2: CRAG EVALUATION RESULTS
================================================================================
Retrieval Corpus:         First {CORPUS_SIZE:,} training passages (train split only)
Retriever:                BM25 + FAISS (RRF k=60)
Top-K Candidates:         {TOP_CANDIDATES}
Relevance Threshold:      >= {RELEVANCE_THRESHOLD}
Fallback:                 Top-{FALLBACK_K} by retrieval rank if none pass
Generator:                {GENERATOR_MODEL}
Evaluator (FP16):         {OUTPUT_DIR}/

Total Questions:          {total_test:,}
Correct Predictions:      {crag_correct:,}
CRAG Accuracy:            {crag_accuracy:.2f}%
Total Time (CRAG):        {crag_elapsed:.2f}s ({crag_elapsed/total_test:.2f}s per question)

--------------------------------------------------------------------------------
FILTERING EFFICIENCY
--------------------------------------------------------------------------------
Total Candidates Evaluated: {total_candidates_eval:,}
Candidates Purged:          {total_purged:,} ({purge_pct:.1f}%)
Candidates Retained:        {total_candidates_eval - total_purged:,} ({100.0 - purge_pct:.1f}%)
================================================================================
"""

print(report)
with open("stage2_detailed_report.txt", "w", encoding="utf-8") as f:
    f.write(report)
logger.info("stage2_detailed_report.txt saved.")
