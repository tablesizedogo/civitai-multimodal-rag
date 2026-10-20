"""Paths and model names. Override paths with environment variables."""
from __future__ import annotations

import os
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parent.parent

# Prebuilt bundle (Quick start): `python -m civitai_search.fetch_prebuilt` downloads the public HF dataset
# (data/meta.jsonl + artifacts/*.npy, no images) into PREBUILT_DIR.
PREBUILT_REPO = os.environ.get("CIVITAI_PREBUILT_REPO", "AER6/civitai-multimodal-rag-data")
PREBUILT_DIR = Path(os.environ.get("CIVITAI_PREBUILT_DIR", PKG_ROOT / "prebuilt")).resolve()

# Which corpus to use:
#   CIVITAI_USE_PREBUILT=1  -> always the prebuilt bundle
#   CIVITAI_USE_PREBUILT=0  -> always the self-collected corpus (full pipeline)
#   unset (auto)            -> self-collected corpus if CIVITAI_DATA_DIR/CIVITAI_ART_DIR are set or ../meta.jsonl
#                              exists, otherwise the prebuilt bundle if it has been fetched.
_explicit = "CIVITAI_DATA_DIR" in os.environ or "CIVITAI_ART_DIR" in os.environ
_mode = os.environ.get("CIVITAI_USE_PREBUILT", "auto").strip().lower()
if _mode in ("1", "true", "yes"):
    USE_PREBUILT = True
elif _mode in ("0", "false", "no"):
    USE_PREBUILT = False
else:
    USE_PREBUILT = (not _explicit and not (PKG_ROOT.parent / "meta.jsonl").exists()
                    and (PREBUILT_DIR / "data" / "meta.jsonl").exists())

# Directory with meta.jsonl and images/ (produced by ../collect.py, or data/ of the prebuilt bundle).
_default_data = PREBUILT_DIR / "data" if USE_PREBUILT else PKG_ROOT.parent
DATA_DIR = Path(os.environ.get("CIVITAI_DATA_DIR", _default_data)).resolve()
META_PATH = DATA_DIR / "meta.jsonl"

# Everything this package computes (embeddings, flags, eval outputs).
_default_art = PREBUILT_DIR / "artifacts" if USE_PREBUILT else PKG_ROOT / "artifacts"
ART_DIR = Path(os.environ.get("CIVITAI_ART_DIR", _default_art)).resolve()

# Images shown in the app: local files from DATA_DIR/images/ when present, otherwise the Civitai CDN thumbnail
# (width=512) stored in meta.jsonl. CIVITAI_REMOTE_IMAGES=1 forces the CDN even if local files exist.
REMOTE_IMAGES = os.environ.get("CIVITAI_REMOTE_IMAGES", "0").strip().lower() in ("1", "true", "yes")

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
