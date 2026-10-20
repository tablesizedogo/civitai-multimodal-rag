"""Compute and cache embeddings for the whole corpus.

    python -m civitai_search.build_embeddings --what clip   # CLIP image embeddings
    python -m civitai_search.build_embeddings --what text   # e5 embeddings of cleaned prompts
    python -m civitai_search.build_embeddings --what mclip  # optional multilingual CLIP (RU queries)

Outputs (in artifacts/):
    ids.txt                    row order shared by all embedding matrices
    clip_image.npy             float16 [N, 512], L2-normalised
    mclip_image.npy            float16 [N, 512], multilingual CLIP (optional)
    text_e5.npy                float16 [N, 384], L2-normalised
CLIP embedding is resumable: finished chunks are kept in artifacts/<name>_chunks/.
"""
from __future__ import annotations

import argparse
import logging
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image

from .config import art
from .data import load_items
from .textnorm import clean_prompt

log = logging.getLogger("build_embeddings")


def write_ids(items) -> None:
    art("ids.txt").write_text("\n".join(str(it.id) for it in items) + "\n")


def build_clip(batch_size: int = 64, chunk: int = 2048, workers: int = 4, multilingual: bool = False) -> None:
    from .encoders import MCLIP_MODEL, MCLIP_PRETRAINED, get_clip

    items = load_items()
    enc = get_clip(MCLIP_MODEL, MCLIP_PRETRAINED) if multilingual else get_clip()
    stem = "mclip_image" if multilingual else "clip_image"
    log.info("CLIP %s, %d images, torch threads=%d", enc.name, len(items), torch.get_num_threads())
    chunk_dir = art(f"{stem}_chunks", "x").parent
    pool = ThreadPoolExecutor(workers)
    t0 = time.time()
    done = 0
    for start in range(0, len(items), chunk):
        out = chunk_dir / f"{start:06d}.npy"
        sub = items[start : start + chunk]
        if out.exists():
            done += len(sub)
            continue
        parts = []
        for b in range(0, len(sub), batch_size):
            batch = sub[b : b + batch_size]
            # decode + preprocess in threads, the forward pass uses all cores
            tensors = list(pool.map(lambda it: enc.preprocess(Image.open(it.path).convert("RGB")), batch))
            with torch.inference_mode():
                e = enc.model.encode_image(torch.stack(tensors)).float().numpy()
            parts.append(e / np.linalg.norm(e, axis=1, keepdims=True))
        np.save(out, np.concatenate(parts).astype(np.float16))
        done += len(sub)
        el = time.time() - t0
        log.info("clip %d/%d  %.1f img/s  elapsed %.0fs", done, len(items), done / max(el, 1e-9), el)
    mats = [np.load(p) for p in sorted(chunk_dir.glob("*.npy"))]
    emb = np.concatenate(mats)
    assert emb.shape[0] == len(items), emb.shape
    np.save(art(f"{stem}.npy"), emb)
    write_ids(items)
    log.info("saved %s.npy %s in %.0fs", stem, emb.shape, time.time() - t0)


def build_text(batch_size: int = 64) -> None:
    from .encoders import get_text_encoder

    items = load_items()
    enc = get_text_encoder()
    texts = [clean_prompt(it.prompt) or it.prompt for it in items]
    t0 = time.time()
    # sort by length to minimise padding, then restore order
    order = np.argsort([len(t) for t in texts])
    emb = enc.encode_passages([texts[i] for i in order], batch_size=batch_size)
    out = np.empty_like(emb)
    out[order] = emb
    np.save(art("text_e5.npy"), out.astype(np.float16))
    write_ids(items)
    log.info("saved text_e5.npy %s in %.0fs", out.shape, time.time() - t0)


def _old(stem: str):
    """Previous matrix + its row ids (sidecar <stem>_ids.txt, falling back to the shared ids.txt)."""
    side = art(f"{stem}_ids.txt")
    idp = side if side.exists() else art("ids.txt")
    m = art(f"{stem}.npy")
    if not m.exists() or not idp.exists():
        return None, {}
    emb = np.load(m)
    ids = [int(x) for x in idp.read_text().split()]
    assert len(ids) == emb.shape[0], (stem, len(ids), emb.shape)
    return emb, {i: r for r, i in enumerate(ids)}


def build_incremental(stem: str, batch_size: int = 64, chunk: int = 2048) -> None:
    """Embed only images/prompts whose id is not in the previous matrix; rows follow load_items() order."""
    items = load_items()
    old, pos = _old(stem)
    new_items = [it for it in items if it.id not in pos]
    log.info("%s: %d items, %d reused, %d to embed", stem, len(items), len(items) - len(new_items), len(new_items))
    t0 = time.time()
    if stem == "text_e5":
        from .encoders import get_text_encoder
        enc = get_text_encoder()
        texts = [clean_prompt(it.prompt) or it.prompt for it in new_items]
        order = np.argsort([len(t) for t in texts])
        e = enc.encode_passages([texts[i] for i in order], batch_size=batch_size) if texts else np.zeros((0, old.shape[1]), np.float32)
        new = np.empty_like(e); new[order] = e
        new = new.astype(np.float16)
    else:
        from .encoders import MCLIP_MODEL, MCLIP_PRETRAINED, get_clip
        enc = get_clip(MCLIP_MODEL, MCLIP_PRETRAINED) if stem == "mclip_image" else get_clip()
        cdir = art(f"{stem}_inc_chunks", "x").parent
        pool = ThreadPoolExecutor(4)
        parts_all = []
        for start in range(0, len(new_items), chunk):
            sub = new_items[start : start + chunk]
            out = cdir / f"{sub[0].id}_{sub[-1].id}_{len(sub)}.npy"
            if out.exists():
                parts_all.append(np.load(out)); continue
            parts = []
            for b in range(0, len(sub), batch_size):
                batch = sub[b : b + batch_size]
                tensors = list(pool.map(lambda it: enc.preprocess(Image.open(it.path).convert("RGB")), batch))
                with torch.inference_mode():
                    e = enc.model.encode_image(torch.stack(tensors)).float().numpy()
                parts.append(e / np.linalg.norm(e, axis=1, keepdims=True))
            arr = np.concatenate(parts).astype(np.float16)
            np.save(out, arr); parts_all.append(arr)
            done = start + len(sub); el = time.time() - t0
            log.info("%s %d/%d  %.1f img/s  elapsed %.0fs", stem, done, len(new_items), done / max(el, 1e-9), el)
        new = np.concatenate(parts_all) if parts_all else np.zeros((0, old.shape[1]), np.float16)
    dim = new.shape[1] if old is None else old.shape[1]
    emb = np.empty((len(items), dim), np.float16)
    nidx = {it.id: r for r, it in enumerate(new_items)}
    for r, it in enumerate(items):
        emb[r] = old[pos[it.id]] if it.id in pos else new[nidx[it.id]]
    np.save(art(f"{stem}.npy"), emb)
    art(f"{stem}_ids.txt").write_text("\n".join(str(it.id) for it in items) + "\n")
    log.info("saved %s.npy %s, embedded %d new in %.0fs", stem, emb.shape, len(new_items), time.time() - t0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", choices=["clip", "mclip", "text", "all"], default="all")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--incremental", action="store_true",
                    help="reuse rows of existing matrices by id, embed only new items (clip, mclip, text)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    if args.incremental:
        stems = {"clip": ["clip_image"], "mclip": ["mclip_image"], "text": ["text_e5"],
                 "all": ["clip_image", "mclip_image", "text_e5"]}[args.what]
        # pin the previous shared row order before anything rewrites ids.txt
        for st in ["clip_image", "mclip_image", "text_e5"]:
            if art(f"{st}.npy").exists() and not art(f"{st}_ids.txt").exists() and art("ids.txt").exists():
                art(f"{st}_ids.txt").write_text(art("ids.txt").read_text())
        for st in stems:
            build_incremental(st, args.batch_size)
        write_ids(load_items())
        return
    if args.what in ("clip", "all"):
        build_clip(args.batch_size)
    if args.what == "mclip":
        build_clip(args.batch_size, multilingual=True)
    if args.what in ("text", "all"):
        build_text(args.batch_size)


if __name__ == "__main__":
    main()
