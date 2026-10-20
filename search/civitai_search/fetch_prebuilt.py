"""Quick start: download the prebuilt index instead of collecting and embedding 100k images.

    python -m civitai_search.fetch_prebuilt               # ~400 MB: metadata + embeddings
    python -m civitai_search.fetch_prebuilt --models      # + CLIP, mCLIP, e5 and the GGUF LLM (~3.7 GB)

Source: the public Hugging Face dataset https://huggingface.co/datasets/AER6/civitai-multimodal-rag-data
(no token needed). Files go to search/prebuilt/ (CIVITAI_PREBUILT_DIR):

    prebuilt/data/meta.jsonl              97,965 strictly-SFW rows: prompt, settings, Civitai page + CDN thumbnail URL
    prebuilt/artifacts/ids.txt            row order shared by all matrices
    prebuilt/artifacts/clip_image.npy     CLIP ViT-B-32 image embeddings (fp16)
    prebuilt/artifacts/mclip_image.npy    multilingual CLIP image embeddings (fp16)
    prebuilt/artifacts/text_e5.npy        multilingual-e5-small prompt embeddings (fp16)
    prebuilt/artifacts/clean_flags.jsonl  empty: flagged rows were already dropped from the bundle

No images are downloaded. The app detects the bundle automatically (see config.USE_PREBUILT) and shows
thumbnails from the Civitai CDN. A self-collected corpus (../meta.jsonl or CIVITAI_DATA_DIR) takes precedence.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from .config import (CLIP_MODEL, CLIP_PRETRAINED, LLM_FILE, LLM_REPO, PREBUILT_DIR, PREBUILT_REPO, TEXT_MODEL)

FILES = ["README.md", "data/meta.jsonl", "artifacts/ids.txt", "artifacts/clip_image.npy",
         "artifacts/mclip_image.npy", "artifacts/text_e5.npy", "artifacts/clean_flags.jsonl"]


def fetch_bundle(repo: str = PREBUILT_REPO, revision: str | None = None) -> None:
    from huggingface_hub import snapshot_download

    t0 = time.time()
    print(f"Downloading {repo} -> {PREBUILT_DIR}")
    snapshot_download(repo, repo_type="dataset", revision=revision, local_dir=PREBUILT_DIR, allow_patterns=FILES)
    total = sum((PREBUILT_DIR / f).stat().st_size for f in FILES if (PREBUILT_DIR / f).exists())
    print(f"  done in {time.time() - t0:.0f}s, {total / 2**20:.0f} MB")


def verify() -> int:
    """Row counts of meta.jsonl, ids.txt and every embedding matrix must agree."""
    import json

    with open(PREBUILT_DIR / "data" / "meta.jsonl", encoding="utf-8") as f:
        meta_ids = sorted(json.loads(line)["id"] for line in f if line.strip())
    ids = [int(x) for x in (PREBUILT_DIR / "artifacts" / "ids.txt").read_text().split()]
    assert ids == meta_ids, "ids.txt does not match meta.jsonl (rows must be sorted by id)"
    for name in ("clip_image.npy", "mclip_image.npy", "text_e5.npy"):
        m = np.load(PREBUILT_DIR / "artifacts" / name, mmap_mode="r")
        assert m.shape[0] == len(ids), f"{name}: {m.shape[0]} rows != {len(ids)}"
        print(f"  {name:16s} {m.shape} {m.dtype}")
    print(f"  OK: {len(ids):,} indexed images")
    return len(ids)


def fetch_models() -> None:
    """Warm the HF cache so the first query / assistant call does not wait for downloads."""
    import open_clip
    from huggingface_hub import hf_hub_download
    from sentence_transformers import SentenceTransformer

    from .encoders import MCLIP_MODEL, MCLIP_PRETRAINED

    t0 = time.time()
    open_clip.create_model_and_transforms(CLIP_MODEL, pretrained=CLIP_PRETRAINED)
    open_clip.create_model_and_transforms(MCLIP_MODEL, pretrained=MCLIP_PRETRAINED)
    open_clip.get_tokenizer(MCLIP_MODEL)
    SentenceTransformer(TEXT_MODEL)
    hf_hub_download(LLM_REPO, LLM_FILE)
    print(f"  models cached in {time.time() - t0:.0f}s")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=PREBUILT_REPO, help="HF dataset repo id")
    ap.add_argument("--revision", default=None, help="dataset revision (branch, tag or commit)")
    ap.add_argument("--models", action="store_true", help="also pre-download CLIP, mCLIP, e5 and the LLM")
    ap.add_argument("--skip-download", action="store_true", help="only verify an existing bundle")
    args = ap.parse_args()
    if not args.skip_download:
        fetch_bundle(args.repo, args.revision)
    verify()
    if args.models:
        fetch_models()
    print("\nNext:  python -m civitai_search.app --port 7860   (thumbnails load from the Civitai CDN)")


if __name__ == "__main__":
    main()
