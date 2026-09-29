"""
medqa_shared.py

Shared utilities for the MedQA 3-Stage CRAG pipeline.
Imported by stage1.py, stage2.py, and stage3.py.

Provides:
  - simple_tokenize      -- BM25 text tokenizer
  - DenseEmbedder        -- HuggingFace mean-pooling embedder (all-MiniLM-L6-v2)
  - HybridRetriever      -- BM25 + FAISS with Reciprocal Rank Fusion (k=60)
  - build_corpus         -- builds the 10,000-doc training knowledge corpus
  - format_prompt        -- flan-t5 prompt template (context + question + options)
  - parse_prediction     -- robust A/B/C/D letter parser

The retrieval corpus is ALWAYS built from the first 10,000 examples of the
"train" split of GBaker/MedQA-USMLE-4-options. It is NEVER built from the
test split.
"""

import json
import logging
import os
import pickle
import re
import shutil
from typing import Dict, List, Optional

import faiss
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CORPUS_SIZE = 10_000
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DATASET_NAME = "GBaker/MedQA-USMLE-4-options"
TRAIN_SPLIT = "train"


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def simple_tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric words for BM25."""
    return re.findall(r"\w+", text.lower())


# ---------------------------------------------------------------------------
# Dense Embedder
# ---------------------------------------------------------------------------

class DenseEmbedder:
    """Computes dense text embeddings using HuggingFace AutoModel with mean pooling."""

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading embedding model '{model_name}' onto {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def encode(
        self,
        texts: List[str],
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Encodes a list of strings into L2-normalised float32 embeddings."""
        all_embeddings = []
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Computing dense embeddings", leave=False)

        with torch.no_grad():
            for i in iterator:
                batch_texts = texts[i : i + batch_size]
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt",
                ).to(self.device)

                outputs = self.model(**encoded)

                # Mean pooling over token embeddings
                token_embeddings = outputs.last_hidden_state
                attention_mask = (
                    encoded["attention_mask"]
                    .unsqueeze(-1)
                    .expand(token_embeddings.size())
                    .float()
                )
                sum_embeddings = torch.sum(token_embeddings * attention_mask, dim=1)
                sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
                embeddings = sum_embeddings / sum_mask

                # L2-normalise for cosine similarity via inner product in FAISS
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                all_embeddings.append(embeddings.cpu().numpy().astype("float32"))

        return np.vstack(all_embeddings)


# ---------------------------------------------------------------------------
# Hybrid Retriever (BM25 + FAISS, Reciprocal Rank Fusion)
# ---------------------------------------------------------------------------

class HybridRetriever:
    """
    Hybrid retriever combining BM25 keyword search and FAISS dense semantic search,
    fused with Reciprocal Rank Fusion (RRF, k=60).

    Parameters
    ----------
    documents : list of str
        The corpus passages to index.
    embedder : DenseEmbedder
        Pre-initialised dense embedder.
    index_dir : str, optional
        Directory to save/load cached BM25 and FAISS indices.
    k_rrf : int
        RRF constant (default 60 per the CRAG spec).
    corpus_size : int
        Expected corpus size; used to detect stale caches.
    """

    def __init__(
        self,
        documents: List[str],
        embedder: DenseEmbedder,
        index_dir: Optional[str] = None,
        k_rrf: int = 60,
        corpus_size: int = CORPUS_SIZE,
    ):
        self.documents = documents
        self.embedder = embedder
        self.k_rrf = k_rrf
        self.corpus_size = corpus_size
        self.bm25: Optional[BM25Okapi] = None
        self.faiss_index: Optional[faiss.IndexFlatIP] = None

        cache_valid = self._is_cache_valid(index_dir)
        if cache_valid:
            self.load_index(index_dir)
        else:
            if index_dir and os.path.exists(index_dir):
                logger.warning(
                    f"Cache in '{index_dir}' is stale (wrong corpus size). "
                    "Deleting and rebuilding..."
                )
                shutil.rmtree(index_dir)
            self.build_index(index_dir)

    # ------------------------------------------------------------------
    # Cache validation
    # ------------------------------------------------------------------

    def _is_cache_valid(self, index_dir: Optional[str]) -> bool:
        """Return True only if the cache exists AND was built with the expected corpus size."""
        if not index_dir:
            return False
        bm25_path = os.path.join(index_dir, "bm25.pkl")
        faiss_path = os.path.join(index_dir, "faiss.index")
        docs_path = os.path.join(index_dir, "documents.json")
        meta_path = os.path.join(index_dir, "corpus_meta.json")

        if not all(os.path.exists(p) for p in [bm25_path, faiss_path, docs_path]):
            return False

        # Check corpus size from metadata
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                cached_size = meta.get("corpus_size", -1)
                if cached_size != self.corpus_size:
                    logger.warning(
                        f"Cache corpus_size={cached_size} != expected {self.corpus_size}."
                    )
                    return False
            except Exception:
                return False
        else:
            # No metadata file means the cache was built by the old version (2,500 docs)
            return False

        return True

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build_index(self, save_dir: Optional[str] = None):
        """Build BM25 and FAISS indices from the document corpus."""
        logger.info(f"Building BM25 index over {len(self.documents):,} documents...")
        tokenized_corpus = [
            simple_tokenize(doc)
            for doc in tqdm(self.documents, desc="Tokenising corpus", leave=False)
        ]
        self.bm25 = BM25Okapi(tokenized_corpus)

        logger.info(f"Building FAISS index over {len(self.documents):,} documents...")
        embeddings = self.embedder.encode(
            self.documents, batch_size=64, show_progress=True
        )
        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)
        self.faiss_index.add(embeddings)
        logger.info(
            f"FAISS index built with {self.faiss_index.ntotal:,} vectors (dim={dimension})."
        )

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            logger.info(f"Saving indices to '{save_dir}'...")
            with open(os.path.join(save_dir, "bm25.pkl"), "wb") as f:
                pickle.dump(self.bm25, f)
            faiss.write_index(
                self.faiss_index, os.path.join(save_dir, "faiss.index")
            )
            with open(
                os.path.join(save_dir, "documents.json"), "w", encoding="utf-8"
            ) as f:
                json.dump(self.documents, f, ensure_ascii=False)
            # Write metadata so future runs can detect stale caches
            with open(
                os.path.join(save_dir, "corpus_meta.json"), "w", encoding="utf-8"
            ) as f:
                json.dump({"corpus_size": len(self.documents)}, f)
            logger.info("Indices successfully cached on disk.")

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load_index(self, index_dir: str):
        """Load cached BM25 and FAISS indices from disk."""
        logger.info(f"Loading cached indices from '{index_dir}'...")
        with open(
            os.path.join(index_dir, "documents.json"), "r", encoding="utf-8"
        ) as f:
            self.documents = json.load(f)
        with open(os.path.join(index_dir, "bm25.pkl"), "rb") as f:
            self.bm25 = pickle.load(f)
        self.faiss_index = faiss.read_index(
            os.path.join(index_dir, "faiss.index")
        )
        logger.info(
            f"Loaded {self.faiss_index.ntotal:,} vectors and BM25 index successfully."
        )

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 20,
    ) -> List[str]:
        """
        Retrieve top_k documents using Reciprocal Rank Fusion over BM25 and FAISS.

        Parameters
        ----------
        query : str
        top_k : int
            Number of final passages to return.
        candidate_k : int
            Size of the initial candidate pool for each retriever.

        Returns
        -------
        list of str  (length <= top_k)
        """
        # BM25
        query_tokens = simple_tokenize(query)
        bm25_scores = self.bm25.get_scores(query_tokens)
        bm25_top = np.argsort(bm25_scores)[::-1][:candidate_k].tolist()

        # FAISS
        query_vec = self.embedder.encode([query], batch_size=1, show_progress=False)
        _, faiss_top = self.faiss_index.search(query_vec, candidate_k)
        faiss_top = faiss_top[0].tolist()

        # Reciprocal Rank Fusion
        rrf_scores: Dict[int, float] = {}
        for rank, idx in enumerate(bm25_top):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (self.k_rrf + rank + 1)
        for rank, idx in enumerate(faiss_top):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (self.k_rrf + rank + 1)

        sorted_indices = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
        return [self.documents[i] for i in sorted_indices[:top_k]]


# ---------------------------------------------------------------------------
# Corpus Builder
# ---------------------------------------------------------------------------

def build_corpus(train_dataset) -> List[str]:
    """
    Build the shared 10,000-document knowledge corpus from the training split.

    The corpus is ALWAYS taken from the training split -- never from the test split.
    Documents are formatted as:
        "Clinical Scenario: {question}\\nMedical Finding & Answer: {answer}"

    Parameters
    ----------
    train_dataset : HuggingFace Dataset
        The full training split. Will be sliced to the first CORPUS_SIZE examples.

    Returns
    -------
    list of str  (length == min(CORPUS_SIZE, len(train_dataset)))
    """
    num_samples = min(CORPUS_SIZE, len(train_dataset))
    subset = train_dataset.select(range(num_samples))
    logger.info(
        f"Building corpus from first {num_samples:,} training passages "
        f"(NEVER from the test split)."
    )
    documents = []
    for item in subset:
        doc = (
            f"Clinical Scenario: {item['question']}\n"
            f"Medical Finding & Answer: {item['answer']}"
        )
        documents.append(doc)
    logger.info(f"Corpus ready: {len(documents):,} clinical passages.")
    return documents


# ---------------------------------------------------------------------------
# Prompt Formatter
# ---------------------------------------------------------------------------

def format_prompt(
    question: str,
    options: Dict[str, str],
    snippets: List[str],
) -> str:
    """
    Format the input prompt for google/flan-t5-base.

    Context snippets are numbered and prepended before the question and options.
    """
    context_text = "\n".join(
        [f"Snippet {i + 1}: {s}" for i, s in enumerate(snippets)]
    )
    prompt = (
        f"Context:\n{context_text}\n\n"
        f"Question: {question}\n\n"
        f"Options:\n"
        f"(A) {options.get('A', '')}\n"
        f"(B) {options.get('B', '')}\n"
        f"(C) {options.get('C', '')}\n"
        f"(D) {options.get('D', '')}\n\n"
        f"Based on the context, select the correct option letter (A, B, C, or D).\n"
        f"Answer:"
    )
    return prompt


# ---------------------------------------------------------------------------
# Answer Parser
# ---------------------------------------------------------------------------

def parse_prediction(generated_text: str, options: Dict[str, str]) -> str:
    """
    Robustly parse model output to extract the answer letter (A, B, C, or D).

    Tries three strategies in order:
      1. Direct regex match for a letter with optional surrounding punctuation.
      2. Fuzzy match against option text values.
      3. First A-D character found anywhere in the output.
    Falls back to 'A' if nothing matches.
    """
    clean_text = generated_text.strip()

    # 1. Direct letter match: 'A', '(A)', 'A.', 'A)'
    match = re.search(r"\(?([A-D])\)?(?:\.|\b)", clean_text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # 2. Fuzzy match against option values
    clean_lower = clean_text.lower()
    best_opt = None
    best_len = 0
    for letter, opt_val in options.items():
        val_lower = opt_val.lower().strip()
        if (val_lower in clean_lower or clean_lower in val_lower) and len(val_lower) > best_len:
            best_opt = letter
            best_len = len(val_lower)
    if best_opt:
        return best_opt

    # 3. First A-D character in output
    for ch in clean_text.upper():
        if ch in ("A", "B", "C", "D"):
            return ch

    return "A"
