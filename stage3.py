"""
stage3.py

Stage 3: Lightweight Quantized CRAG Evaluation.

Loads the FP16 DistilBERT evaluator checkpoint saved by stage2.py from
./my_medical_evaluator, then quantizes it for efficient inference.

This script auto-detects the compute device at runtime:

GPU path  (CUDA available):
  BitsAndBytesConfig — 4-bit NF4 quantization via bitsandbytes.
  The results reported in this thesis were produced running stage3.py on
  Google Colab with a Tesla T4 GPU, using this bitsandbytes NF4 path.

CPU path  (no CUDA):
  torch.quantization.quantize_dynamic — int8 dynamic quantization.
  A clear warning is printed and the fallback is recorded in the report.
  The CPU fallback exists so the script remains runnable on a machine
  without a GPU, but was NOT the path used for the final reported numbers.

The retrieval corpus is ALWAYS built from the first 10,000 training passages
(never the test split). Uses the same HybridRetriever, format_prompt, and
parse_prediction as stage1 and stage2 (imported from medqa_shared).

CRAG loop is identical to stage2:
  - Top-5 candidate retrieval
  - Score with quantized evaluator; keep >= 0.50 relevance
  - Fallback to top-2 by retrieval rank if nothing passes
  - Generate with google/flan-t5-base
  - Parse answer letter

Outputs:
  - stage3_detailed_report.txt
  - stage3_predictions.json
"""

import json
import logging
import os
import time

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
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

EVALUATOR_DIR = "./my_medical_evaluator"
GENERATOR_MODEL = "google/flan-t5-base"
DATASET_NAME = "GBaker/MedQA-USMLE-4-options"
INDEX_DIR = ".cache_index"
RELEVANCE_THRESHOLD = 0.50
TOP_CANDIDATES = 5
FALLBACK_K = 2
MAX_LENGTH = 256

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("MedQA-Stage3")


def get_model_size_mb(model) -> float:
    """Return approximate model size in MB (parameter bytes only)."""
    total_bytes = sum(
        p.nelement() * p.element_size() for p in model.parameters()
    )
    return total_bytes / (1024 ** 2)


def main():
    # ------------------------------------------------------------------
    # 1. Device check
    # ------------------------------------------------------------------
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")
    logger.info(f"Target compute device: {device}")

    if not cuda_available:
        print(
            "\n" + "!" * 70 + "\n"
            "WARNING: No NVIDIA GPU / CUDA detected.\n"
            "4-bit NF4 quantization via bitsandbytes requires an NVIDIA GPU with CUDA.\n"
            "Falling back to torch.quantization.quantize_dynamic (int8, CPU-compatible).\n"
            + "!" * 70 + "\n"
        )
    else:
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # ------------------------------------------------------------------
    # 2. Verify stage2 checkpoint exists
    # ------------------------------------------------------------------
    if not os.path.isdir(EVALUATOR_DIR):
        raise FileNotFoundError(
            f"Stage2 evaluator checkpoint not found at '{EVALUATOR_DIR}'. "
            "Please run stage2.py first."
        )
    logger.info(f"Loading FP16 evaluator checkpoint from '{EVALUATOR_DIR}'...")

    # ------------------------------------------------------------------
    # 3. Measure pre-quantization model size
    # ------------------------------------------------------------------
    base_model = AutoModelForSequenceClassification.from_pretrained(
        EVALUATOR_DIR, torch_dtype=torch.float16
    )
    size_before_mb = get_model_size_mb(base_model)
    logger.info(f"Evaluator size before quantization: {size_before_mb:.1f} MB")
    del base_model  # free memory before quantization load
    if cuda_available:
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 4. Quantize
    # ------------------------------------------------------------------
    if cuda_available:
        # --- 4-bit NF4 via bitsandbytes ---
        quantization_method = "bitsandbytes NF4 (4-bit)"
        logger.info("Applying 4-bit NF4 quantization via bitsandbytes...")

        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        eval_model = AutoModelForSequenceClassification.from_pretrained(
            EVALUATOR_DIR,
            quantization_config=quantization_config,
            device_map="auto",
        )
        eval_model.eval()
        # bitsandbytes stores weights in 4-bit; reported size is based on active params
        size_after_mb = get_model_size_mb(eval_model)
        logger.info(
            f"Quantization complete (NF4): "
            f"{size_before_mb:.1f} MB -> {size_after_mb:.1f} MB"
        )
        use_amp = True

    else:
        # --- int8 dynamic quantization via torch.quantization ---
        quantization_method = "torch.quantization.quantize_dynamic (int8, CPU)"
        logger.info("Applying dynamic int8 quantization (CPU fallback)...")

        fp32_model = AutoModelForSequenceClassification.from_pretrained(
            EVALUATOR_DIR, torch_dtype=torch.float32
        )
        fp32_model.eval()
        eval_model = torch.quantization.quantize_dynamic(
            fp32_model,
            {torch.nn.Linear},
            dtype=torch.qint8,
        )
        size_after_mb = get_model_size_mb(eval_model)
        logger.info(
            f"Quantization complete (int8): "
            f"{size_before_mb:.1f} MB -> {size_after_mb:.1f} MB"
        )
        use_amp = False

    compression_ratio = size_before_mb / max(size_after_mb, 0.001)
    logger.info(f"Compression ratio: {compression_ratio:.1f}x")

    eval_tokenizer = AutoTokenizer.from_pretrained(EVALUATOR_DIR)

    # ------------------------------------------------------------------
    # 5. Load generator
    # ------------------------------------------------------------------
    logger.info(f"Loading generator model '{GENERATOR_MODEL}'...")
    gen_tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL)
    gen_model = AutoModelForSeq2SeqLM.from_pretrained(GENERATOR_MODEL).to(device)
    gen_model.eval()

    # ------------------------------------------------------------------
    # 6. Build corpus and retriever (TRAINING split only, never test)
    # ------------------------------------------------------------------
    logger.info("Loading training split for corpus construction...")
    train_raw = load_dataset(DATASET_NAME, split="train")
    documents = build_corpus(train_raw)  # first 10,000 training passages

    embedder = DenseEmbedder(device=str(device))
    retriever = HybridRetriever(
        documents=documents,
        embedder=embedder,
        index_dir=INDEX_DIR,
        k_rrf=60,
        corpus_size=CORPUS_SIZE,
    )

    # ------------------------------------------------------------------
    # 7. Load test split (questions only — never added to corpus)
    # ------------------------------------------------------------------
    logger.info("Loading MedQA test split (1,273 questions)...")
    test_dataset = load_dataset(DATASET_NAME, split="test")
    total_test = len(test_dataset)
    logger.info(f"Test questions: {total_test:,}")

    # ------------------------------------------------------------------
    # 8. CRAG evaluation loop
    # ------------------------------------------------------------------
    logger.info(f"Starting Stage 3 CRAG evaluation on {total_test:,} questions...")
    correct_count = 0
    total_candidates_eval = 0
    total_purged = 0
    results = []
    start_time = time.time()

    for item in tqdm(test_dataset, desc="Stage3 CRAG Eval", unit="q"):
        question = item["question"]
        options = item["options"]
        ground_truth = item.get("answer_idx") or item.get("answer")

        # --- Retrieve top-5 candidates ---
        candidates = retriever.retrieve(question, top_k=TOP_CANDIDATES)

        # --- Score with quantized evaluator ---
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
                if use_amp and cuda_available:
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        outputs = eval_model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                        )
                else:
                    outputs = eval_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                probs = torch.softmax(outputs.logits.float(), dim=-1)
                relevance_score = probs[0][1].item()

            relevance_scores.append(relevance_score)
            if relevance_score >= RELEVANCE_THRESHOLD:
                retained_snippets.append(cand)
            else:
                total_purged += 1

        # --- Fallback to top-2 by retrieval rank ---
        if not retained_snippets:
            retained_snippets = candidates[:FALLBACK_K]

        # --- Generate answer ---
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

        # --- Resolve ground truth ---
        target_letter = str(ground_truth).strip().upper()
        if target_letter not in ("A", "B", "C", "D"):
            for opt_k, opt_v in options.items():
                if opt_v.strip() == str(ground_truth).strip():
                    target_letter = opt_k
                    break

        is_correct = predicted_letter == target_letter
        if is_correct:
            correct_count += 1

        results.append({
            "question": question,
            "options": options,
            "target": target_letter,
            "prediction": predicted_letter,
            "raw_generation": raw_gen,
            "is_correct": is_correct,
            "retrieved_snippets": candidates,
            "retained_snippets": retained_snippets,
            "relevance_scores": relevance_scores,
            "quantization_method": quantization_method,
        })

    elapsed_time = time.time() - start_time
    final_accuracy = (correct_count / total_test) * 100.0
    purge_pct = (total_purged / max(total_candidates_eval, 1)) * 100.0

    logger.info(
        f"Stage 3 Complete! Accuracy: {final_accuracy:.2f}% "
        f"({correct_count}/{total_test})"
    )

    # ------------------------------------------------------------------
    # 9. Save predictions JSON
    # ------------------------------------------------------------------
    predictions_summary = {
        "dataset_name": DATASET_NAME,
        "test_split": "test",
        "total_questions": total_test,
        "correct_predictions": correct_count,
        "crag_accuracy_percentage": round(final_accuracy, 2),
        "time_elapsed_seconds": round(elapsed_time, 2),
        "generator_model": GENERATOR_MODEL,
        "evaluator_checkpoint": EVALUATOR_DIR,
        "quantization_method": quantization_method,
        "model_size_before_mb": round(size_before_mb, 1),
        "model_size_after_mb": round(size_after_mb, 1),
        "compression_ratio": round(compression_ratio, 2),
        "relevance_threshold": RELEVANCE_THRESHOLD,
        "top_candidates": TOP_CANDIDATES,
        "fallback_k": FALLBACK_K,
        "total_candidates_evaluated": total_candidates_eval,
        "total_candidates_purged": total_purged,
        "detailed_results": results,
    }

    with open("stage3_predictions.json", "w", encoding="utf-8") as f:
        json.dump(predictions_summary, f, indent=2, ensure_ascii=False)
    logger.info("stage3_predictions.json saved.")

    # ------------------------------------------------------------------
    # 10. Write report
    # ------------------------------------------------------------------
    gpu_label = torch.cuda.get_device_name(0) if cuda_available else "CPU (no CUDA)"

    report_content = f"""================================================================================
       MEDQA STAGE 3: QUANTIZED CRAG EVALUATION REPORT
================================================================================
Dataset:                    {DATASET_NAME} (test split)
Evaluator Checkpoint:       {EVALUATOR_DIR}/ (trained by stage2.py)
Generator Model:            {GENERATOR_MODEL}
Compute Device:             {gpu_label}

--------------------------------------------------------------------------------
QUANTIZATION DETAILS
--------------------------------------------------------------------------------
Method Used:                {quantization_method}
Model Size (before):        {size_before_mb:.1f} MB  (FP16 checkpoint)
Model Size (after):         {size_after_mb:.1f} MB
Compression Ratio:          {compression_ratio:.1f}x smaller

--------------------------------------------------------------------------------
RETRIEVAL CONFIGURATION
--------------------------------------------------------------------------------
Corpus Source:              First {CORPUS_SIZE:,} training passages (train split ONLY)
Retriever:                  BM25 + FAISS (RRF k=60)
Top-K Candidates:           {TOP_CANDIDATES}
Relevance Threshold:        >= {RELEVANCE_THRESHOLD}
Fallback Policy:            Top-{FALLBACK_K} by retrieval rank (if none pass threshold)

--------------------------------------------------------------------------------
FILTERING EFFICIENCY
--------------------------------------------------------------------------------
Total Candidates Evaluated: {total_candidates_eval:,}
Candidates Purged:          {total_purged:,} ({purge_pct:.1f}%)
Candidates Retained:        {total_candidates_eval - total_purged:,} ({100.0 - purge_pct:.1f}%)

--------------------------------------------------------------------------------
CRAG EVALUATION RESULTS
--------------------------------------------------------------------------------
Total Questions:            {total_test:,}
Correct Predictions:        {correct_count:,}
Final CRAG Accuracy:        {final_accuracy:.2f}%
Total Time Elapsed:         {elapsed_time:.2f}s ({elapsed_time/total_test:.2f}s per question)
================================================================================
"""

    with open("stage3_detailed_report.txt", "w", encoding="utf-8") as f:
        f.write(report_content)
    logger.info("stage3_detailed_report.txt saved.")
    print(report_content)


if __name__ == "__main__":
    main()
