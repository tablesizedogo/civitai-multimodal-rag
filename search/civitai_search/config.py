"""Paths and model names. Override paths with environment variables."""
from __future__ import annotations

import os
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parent.parent

# Directory with meta.jsonl and images/ (produced by a separate collector).
DATA_DIR = Path(os.environ.get("CIVITAI_DATA_DIR", PKG_ROOT.parent)).resolve()
META_PATH = DATA_DIR / "meta.jsonl"

# Everything this package computes (embeddings, flags, eval outputs).
ART_DIR = Path(os.environ.get("CIVITAI_ART_DIR", PKG_ROOT / "artifacts")).resolve()

CLIP_MODEL = os.environ.get("CLIP_MODEL", "ViT-B-32")
CLIP_PRETRAINED = os.environ.get("CLIP_PRETRAINED", "laion2b_s34b_b79k")
TEXT_MODEL = os.environ.get("TEXT_MODEL", "intfloat/multilingual-e5-small")


def art(*parts: str) -> Path:
    p = ART_DIR.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

# Local LLM for the generation step (GGUF, run with llama.cpp on CPU).
# LLM_GGUF may point to a local file; otherwise LLM_REPO/LLM_FILE is fetched from the HF Hub.
LLM_REPO = os.environ.get("LLM_REPO", "Qwen/Qwen2.5-1.5B-Instruct-GGUF")
LLM_FILE = os.environ.get("LLM_FILE", "qwen2.5-1.5b-instruct-q4_k_m.gguf")
LLM_GGUF = os.environ.get("LLM_GGUF", "")
LLM_THREADS = int(os.environ.get("LLM_THREADS", os.cpu_count() or 4))
