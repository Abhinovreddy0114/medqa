"""
stage1.py

Complete baseline evaluation pipeline for MedQA (USMLE 4-options).
1. Loads MedQA test split (1,273 questions) from Hugging Face.
2. Builds BM25 keyword search and FAISS semantic search over a 10,000-document
   medical knowledge corpus drawn from the TRAINING split (never the test split).
3. Retrieves relevant snippets using hybrid search (Reciprocal Rank Fusion, k=60).
4. Passes retrieved snippets + question + options to google/flan-t5-base.
5. Evaluates predictions and prints the final baseline accuracy percentage.

Shared retrieval utilities (DenseEmbedder, HybridRetriever, build_corpus,
format_prompt, parse_prediction) are imported from medqa_shared.py.
"""

import argparse
import json
import logging
import os
import time
from typing import Dict, List

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from medqa_shared import (
    CORPUS_SIZE,
    DenseEmbedder,
    HybridRetriever,
    build_corpus,
    format_prompt,
    parse_prediction,
)

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("MedQA-Stage1")


# ------------------------------------------------------------------------------
# Main Evaluation Pipeline
# ------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="MedQA Stage 1 Baseline Evaluation (BM25 + FAISS + Flan-T5)"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="GBaker/MedQA-USMLE-4-options",
        help="Hugging Face dataset name for MedQA",
    )
    parser.add_argument(
        "--test_split",
        type=str,
        default="test",
        help="Split name for testing (1,273 questions)",
    )
    parser.add_argument(
        "--corpus_split",
        type=str,
        default="train",
        help="Hugging Face split name used as the reference knowledge corpus",
    )
    parser.add_argument(
        "--max_corpus_samples",
        type=int,
        default=10000,
        help="Maximum number of corpus passages to index (default: 10,000)",
    )
    parser.add_argument(
        "--generator_model",
        type=str,
        default="google/flan-t5-base",
        help="Generator model name",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model for FAISS semantic search",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Number of retrieved snippets to pass to generator",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Inference batch size for Flan-T5 generator",
    )
    parser.add_argument(
        "--index_dir",
        type=str,
        default=".cache_index",
        help="Directory to save/load cached BM25 and FAISS indices",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of test questions evaluated (None = all 1,273 questions)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="stage1_predictions.json",
        help="Path to save detailed predictions and evaluation metrics JSON",
    )
    parser.add_argument(
        "--report_file",
        type=str,
        default="stage1_detailed_report.txt",
        help="Path to save human-readable detailed evaluation report log file",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using compute device: {device}")

    # 1. Load MedQA Test Split from Hugging Face
    logger.info(
        f"Loading MedQA test split from Hugging Face "
        f"('{args.dataset_name}', split='{args.test_split}')..."
    )
    test_dataset = load_dataset(args.dataset_name, split=args.test_split)
    total_test_samples = len(test_dataset)
    logger.info(f"Loaded {total_test_samples:,} questions from test split.")

    if args.limit:
        eval_samples = min(args.limit, total_test_samples)
        test_dataset = test_dataset.select(range(eval_samples))
        logger.info(f"Evaluating subset of {eval_samples} questions (--limit).")
    else:
        eval_samples = total_test_samples

    # 2. Prepare Knowledge Corpus for BM25 and FAISS
    #    Always from the TRAINING split, first 10,000 examples.
    logger.info(
        f"Loading reference medical corpus from '{args.dataset_name}' "
        f"(split='{args.corpus_split}')..."
    )
    train_dataset = load_dataset(args.dataset_name, split=args.corpus_split)
    # build_corpus slices to CORPUS_SIZE (10,000) internally
    documents = build_corpus(train_dataset)

    # 3. Initialize Embedder and Hybrid Retriever
    embedder = DenseEmbedder(model_name=args.embedding_model, device=device)
    retriever = HybridRetriever(
        documents=documents,
        embedder=embedder,
        index_dir=args.index_dir,
        k_rrf=60,
        corpus_size=CORPUS_SIZE,
    )

    # 4. Load Flan-T5 Generator
    logger.info(f"Loading generator model '{args.generator_model}' onto {device}...")
    gen_tokenizer = AutoTokenizer.from_pretrained(args.generator_model)
    gen_model = AutoModelForSeq2SeqLM.from_pretrained(args.generator_model).to(device)
    gen_model.eval()

    # 5. Run Evaluation through the Test Split
    logger.info(f"Starting evaluation across {eval_samples:,} MedQA questions...")
    correct_count = 0
    results = []
    start_time = time.time()

    batch_size = args.batch_size
    pbar = tqdm(total=eval_samples, desc="Evaluating MedQA", unit="q")

    for i in range(0, eval_samples, batch_size):
        batch_items = [test_dataset[j] for j in range(i, min(i + batch_size, eval_samples))]

        batch_prompts = []
        batch_metadata = []

        for item in batch_items:
            question = item["question"]
            options = item["options"]
            ground_truth = item.get("answer_idx") or item.get("answer")

            # Retrieve top_k relevant snippets using hybrid BM25 + FAISS (RRF)
            retrieved_snippets = retriever.retrieve(question, top_k=args.top_k)

            prompt = format_prompt(question, options, retrieved_snippets)
            batch_prompts.append(prompt)
            batch_metadata.append({
                "question": question,
                "options": options,
                "ground_truth": ground_truth,
                "retrieved_snippets": retrieved_snippets,
                "prompt": prompt,
            })

        # Tokenize and generate with Flan-T5
        inputs = gen_tokenizer(
            batch_prompts,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = gen_model.generate(
                **inputs,
                max_new_tokens=16,
                num_beams=1,
                do_sample=False,
            )

        # Decode and score predictions
        for meta, out_ids in zip(batch_metadata, outputs):
            raw_gen = gen_tokenizer.decode(out_ids, skip_special_tokens=True)
            predicted_letter = parse_prediction(raw_gen, meta["options"])

            target_letter = str(meta["ground_truth"]).strip().upper()
            if target_letter not in ("A", "B", "C", "D"):
                for opt_k, opt_v in meta["options"].items():
                    if opt_v.strip() == str(meta["ground_truth"]).strip():
                        target_letter = opt_k
                        break

            is_correct = predicted_letter == target_letter
            if is_correct:
                correct_count += 1

            results.append({
                "question": meta["question"],
                "options": meta["options"],
                "target": target_letter,
                "prediction": predicted_letter,
                "raw_generation": raw_gen,
                "is_correct": is_correct,
                "retrieved_snippets": meta["retrieved_snippets"],
            })

            pbar.update(1)
            running_acc = (correct_count / len(results)) * 100.0
            pbar.set_postfix({
                "Accuracy": f"{running_acc:.2f}%",
                "Correct": f"{correct_count}/{len(results)}",
            })

    pbar.close()
    elapsed_time = time.time() - start_time

    # 6. Calculate & Print Final Baseline Accuracy
    final_accuracy = (correct_count / eval_samples) * 100.0

    print("\n" + "=" * 70)
    print("           MEDQA STAGE 1 BASELINE EVALUATION REPORT           ")
    print("=" * 70)
    print(f" Dataset Name:              {args.dataset_name}")
    print(f" Split Evaluated:           {args.test_split}")
    print(f" Total Questions:           {eval_samples:,}")
    print(f" Correct Predictions:       {correct_count:,}")
    print(f" Final Baseline Accuracy:   {final_accuracy:.2f}%")
    print(f" Total Time Elapsed:        {elapsed_time:.2f}s ({elapsed_time/eval_samples:.2f}s per question)")
    print(f" Device Used:               {device}")
    print(f" Generator Model:           {args.generator_model}")
    print(f" Hybrid Retrieval:          BM25 + FAISS (RRF top-{args.top_k})")
    print("=" * 70)

    # Save detailed predictions JSON
    summary = {
        "dataset_name": args.dataset_name,
        "test_split": args.test_split,
        "total_questions": eval_samples,
        "correct_predictions": correct_count,
        "baseline_accuracy_percentage": round(final_accuracy, 2),
        "time_elapsed_seconds": round(elapsed_time, 2),
        "generator_model": args.generator_model,
        "embedding_model": args.embedding_model,
        "retrieval": f"BM25 + FAISS (top_{args.top_k})",
        "detailed_results": results,
    }

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Detailed evaluation results saved to '{args.output_file}'.")

    # Save human-readable report
    report_lines = [
        "=" * 80,
        "                    MEDQA STAGE 1 DETAILED EVALUATION REPORT",
        "=" * 80,
        f"Dataset:                  {args.dataset_name} ({args.test_split} split)",
        f"Generator Model:          {args.generator_model}",
        f"Embedding Model:          {args.embedding_model}",
        f"Retrieval Method:         BM25 + FAISS (RRF top-{args.top_k})",
        f"Corpus Size:              {CORPUS_SIZE:,} training passages (train split only)",
        f"Total Questions Run:      {eval_samples:,}",
        f"Correct Predictions:      {correct_count:,}",
        f"Final Baseline Accuracy:  {final_accuracy:.2f}%",
        f"Total Time Elapsed:       {elapsed_time:.2f}s ({elapsed_time / eval_samples:.2f}s per question)",
        "=" * 80,
        "",
        "-" * 80,
        "SAMPLE QUESTIONS BREAKDOWN (5 SAMPLES)",
        "-" * 80,
    ]

    num_samples_to_show = min(5, len(results))
    for idx in range(num_samples_to_show):
        res = results[idx]
        status_str = "CORRECT" if res["is_correct"] else "INCORRECT"
        report_lines.extend([
            f"\n[Sample Question {idx + 1}] - Result: {status_str}",
            "-" * 40,
            f"Question:\n{res['question']}\n",
            "Options:",
            f"  (A) {res['options'].get('A', '')}",
            f"  (B) {res['options'].get('B', '')}",
            f"  (C) {res['options'].get('C', '')}",
            f"  (D) {res['options'].get('D', '')}\n",
            "Retrieved Context Snippets:",
        ])
        for s_idx, snip in enumerate(res["retrieved_snippets"]):
            report_lines.append(f"  --- Snippet {s_idx + 1} ---\n  {snip}\n")
        report_lines.extend([
            f"Ground Truth Target: {res['target']}",
            f"Generated Answer:    {res['prediction']} (Raw output: '{res['raw_generation']}')",
            f"Evaluation:          {status_str}",
            "-" * 80,
        ])

    report_content = "\n".join(report_lines) + "\n"
    with open(args.report_file, "w", encoding="utf-8") as f:
        f.write(report_content)
    logger.info(f"Detailed human-readable report saved to '{args.report_file}'.")


if __name__ == "__main__":
    main()
