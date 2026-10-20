"""Four retrievers over the same corpus.

A  BM25Retriever      - lexical search over cleaned prompt text
B  PromptEmbRetriever - multilingual-e5 embeddings of cleaned prompts (FAISS)
C  ClipRetriever      - CLIP image embeddings (FAISS); text->image and image->image
D  HybridRetriever    - reciprocal rank fusion of B and C
E  ClipRetriever(multilingual=True) - XLM-R CLIP, for Russian queries (optional)
F  HybridRetriever    - RRF of B and E (optional)

All retrievers return a list of (row_index, score) for rows of `load_items()`.
Rows flagged in clean_flags.jsonl (text-only junk, near-duplicates) can be
excluded by passing `keep` (array of row indices allowed in results).
"""
from __future__ import annotations

import json
from collections import Counter

import faiss
import numpy as np
from PIL import Image
from scipy import sparse

from .config import art
from .data import load_items
from .textnorm import clean_prompt, tokenize

Hit = tuple[int, float]


def load_keep_mask(exclude_flags: bool = True) -> np.ndarray:
    """Row indices allowed in search results."""
    n = len(load_items())
    path = art("clean_flags.jsonl")
    if not exclude_flags or not path.exists():
        return np.arange(n)
    bad = set()
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("text_junk") or r.get("dup_of") is not None:
                bad.add(r["row"])
    return np.array([i for i in range(n) if i not in bad])


class BM25Retriever:
    name = "A: BM25 (prompt)"

    def __init__(self, keep: np.ndarray, k1: float = 1.2, b: float = 0.75):
        items = load_items()
        self.rows = np.asarray(keep)
        docs = [tokenize(clean_prompt(items[i].prompt)) for i in self.rows]
        vocab: dict[str, int] = {}
        r_idx, c_idx, tf = [], [], []
        for d, toks in enumerate(docs):
            for tok, c in Counter(toks).items():
                r_idx.append(d)
                c_idx.append(vocab.setdefault(tok, len(vocab)))
                tf.append(c)
        tf = np.asarray(tf, dtype=np.float32)
        r_idx = np.asarray(r_idx)
        c_idx = np.asarray(c_idx)
        dl = np.array([len(t) for t in docs], dtype=np.float32)
        avgdl = dl.mean()
        df = np.bincount(c_idx, minlength=len(vocab))
        n = len(docs)
        idf = np.log(1 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
        w = idf[c_idx] * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl[r_idx] / avgdl))
        # term-major matrix: scoring a query = summing a few rows
        self.mat = sparse.csr_matrix((w, (c_idx, r_idx)), shape=(len(vocab), n))
        self.vocab = vocab

    def search(self, query: str, k: int = 10) -> list[Hit]:
        terms = [self.vocab[t] for t in tokenize(query) if t in self.vocab]
        if not terms:
            return []
        scores = np.asarray(self.mat[terms].sum(axis=0)).ravel()
        k = min(k, int((scores > 0).sum()))
        if k == 0:
            return []
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(self.rows[i]), float(scores[i])) for i in top]


class _FaissIndex:
    def __init__(self, emb: np.ndarray, keep: np.ndarray):
        self.rows = np.asarray(keep)
        x = np.ascontiguousarray(emb[self.rows].astype(np.float32))
        self.index = faiss.IndexFlatIP(x.shape[1])
        self.index.add(x)

    def search(self, q: np.ndarray, k: int) -> list[Hit]:
        d, i = self.index.search(np.ascontiguousarray(q.reshape(1, -1).astype(np.float32)), k)
        return [(int(self.rows[j]), float(s)) for j, s in zip(i[0], d[0]) if j >= 0]


class PromptEmbRetriever:
    name = "B: e5 (prompt)"

    def __init__(self, keep: np.ndarray):
        from .encoders import get_text_encoder

        self.enc = get_text_encoder()
        self.index = _FaissIndex(np.load(art("text_e5.npy")), keep)

    def search(self, query: str, k: int = 10) -> list[Hit]:
        return self.index.search(self.enc.encode_queries([query])[0], k)


class ClipRetriever:
    name = "C: CLIP (image)"

    def __init__(self, keep: np.ndarray, multilingual: bool = False):
        from .encoders import MCLIP_MODEL, MCLIP_PRETRAINED, get_clip

        if multilingual:
            self.enc = get_clip(MCLIP_MODEL, MCLIP_PRETRAINED)
            emb_file, self.name = "mclip_image.npy", "E: mCLIP (image)"
        else:
            self.enc = get_clip()
            emb_file = "clip_image.npy"
        self.emb = np.load(art(emb_file))
        self.index = _FaissIndex(self.emb, keep)

    def search(self, query: str, k: int = 10) -> list[Hit]:
        return self.index.search(self.enc.encode_texts([query])[0], k)

    def search_image(self, image: Image.Image, k: int = 10) -> list[Hit]:
        return self.index.search(self.enc.encode_images([image])[0], k)


class HybridRetriever:
    """Reciprocal rank fusion: score = sum 1 / (rrf_k + rank)."""

    name = "D: hybrid RRF (B+C)"

    def __init__(self, parts: list, rrf_k: int = 60, depth: int = 100, name: str | None = None):
        self.parts, self.rrf_k, self.depth = parts, rrf_k, depth
        if name:
            self.name = name

    def search(self, query: str, k: int = 10) -> list[Hit]:
        fused: dict[int, float] = {}
        for p in self.parts:
            for rank, (row, _) in enumerate(p.search(query, self.depth)):
                fused[row] = fused.get(row, 0.0) + 1.0 / (self.rrf_k + rank + 1)
        return sorted(fused.items(), key=lambda x: -x[1])[:k]


def build_all(exclude_flags: bool = True, multilingual: bool | None = None,
              only: str | None = None) -> dict[str, object]:
    """Build retrievers A-F (E/F only if mclip_image.npy exists).

    `only` is a string of method keys (e.g. "EF"); dependencies are built
    automatically (F needs B and E) and only requested methods are returned.
    Loading just what is needed keeps RAM low (each CLIP model + index ~0.6 GB).
    """
    keep = load_keep_mask(exclude_flags)
    if multilingual is None:
        multilingual = art("mclip_image.npy").exists()
    want = set(only or "ABCDEF")
    if not multilingual:
        want -= {"E", "F"}
    need = set(want)
    if "D" in need:
        need |= {"B", "C"}
    if "F" in need:
        need |= {"B", "E"}
    rets: dict[str, object] = {}
    if "A" in need:
        rets["A"] = BM25Retriever(keep)
    if "B" in need:
        rets["B"] = PromptEmbRetriever(keep)
    if "C" in need:
        rets["C"] = ClipRetriever(keep)
    if "D" in need:
        rets["D"] = HybridRetriever([rets["B"], rets["C"]])
    if "E" in need:
        rets["E"] = ClipRetriever(keep, multilingual=True)
    if "F" in need:
        rets["F"] = HybridRetriever([rets["B"], rets["E"]], name="F: hybrid RRF (B+E)")
    return {k: rets[k] for k in "ABCDEF" if k in rets and (k in want or only is None)}
