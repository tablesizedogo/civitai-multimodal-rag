"""Flag (never delete) junk and near-duplicate images.

    python -m civitai_search.clean [--junk-hi 0.9 --junk-lo 0.4] [--dup-threshold 0.97]

* text_junk: CLIP zero-shot says the picture is mostly text (banners like
  "thanks for stopping by, leave a reaction"). Score = softmax probability mass
  of the "text" prompts against a set of ordinary-image prompts. Flag if
  score > 0.9, or score > 0.4 and the prompt asks for buzz/reactions/tips
  (thresholds picked by eyeballing the grids; no OCR on this box).
* dup_of: cosine similarity of CLIP image embeddings > threshold to an
  earlier row in the same duplicate group (union-find); the first row of a
  group is kept as the representative.

Writes artifacts/clean_flags.jsonl (one line per flagged row) and
artifacts/clean_junk_grid.png / clean_dup_grid.png for eyeballing.
"""
from __future__ import annotations

import argparse
import json

import re

import faiss
import numpy as np

from .config import art
from .data import load_items
from .encoders import get_clip
from .viz import image_grid

JUNK_PROMPTS = [
    "a screenshot of text",
    "a poster with only text on it",
    "a banner with a text message",
    "a page of written text",
    "a thank you message card with text",
    "large caption text on a plain background",
    "a logo with text",
]
NORMAL_PROMPTS = [
    "a photo",
    "a digital painting",
    "an anime illustration",
    "a portrait of a person",
    "a landscape",
    "a 3d render",
    "a cartoon character",
    "an illustration of an animal",
    "a fantasy artwork",
    "a picture of food",
    "a building or interior",
    "a poster with a character and a title",
]


# Prompts of "give me buzz / thanks for the reaction" banners on Civitai.
BEGGING = re.compile(r"\b(buzz|thank(?:s| you)|reaction|react|tip(?:ping)?|like and|follow me|subscribe)\b", re.I)


def is_junk(score: float, prompt: str, hi: float, lo: float) -> bool:
    """Very confident CLIP text score, or a weaker score plus a begging prompt."""
    return score > hi or (score > lo and bool(BEGGING.search(prompt)))


def junk_scores(emb: np.ndarray) -> np.ndarray:
    enc = get_clip()
    t = enc.encode_texts(JUNK_PROMPTS + NORMAL_PROMPTS)
    logits = 100.0 * emb.astype(np.float32) @ t.T
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    p /= p.sum(axis=1, keepdims=True)
    return p[:, : len(JUNK_PROMPTS)].sum(axis=1)


def dup_groups(emb: np.ndarray, threshold: float, k: int = 10) -> dict[int, int]:
    """Map row -> representative row for every non-representative duplicate."""
    x = np.ascontiguousarray(emb.astype(np.float32))
    index = faiss.IndexFlatIP(x.shape[1])
    index.add(x)
    sims, nbrs = index.search(x, k)
    parent = list(range(len(x)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(len(x)):
        for s, j in zip(sims[i], nbrs[i]):
            if j != i and j >= 0 and s > threshold:
                ra, rb = find(i), find(int(j))
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
    return {i: find(i) for i in range(len(x)) if find(i) != i}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--junk-hi", type=float, default=0.9, help="flag on CLIP score alone")
    ap.add_argument("--junk-lo", type=float, default=0.4, help="flag if prompt also looks like begging")
    ap.add_argument("--dup-threshold", type=float, default=0.97)
    args = ap.parse_args()

    items = load_items()
    emb = np.load(art("clip_image.npy"))
    js = junk_scores(emb)
    np.save(art("junk_scores.npy"), js.astype(np.float32))
    dups = dup_groups(emb, args.dup_threshold)

    flagged = []
    for i, it in enumerate(items):
        junk = is_junk(float(js[i]), it.prompt, args.junk_hi, args.junk_lo)
        if junk or i in dups:
            flagged.append(
                {
                    "row": i,
                    "id": it.id,
                    "text_junk": junk,
                    "junk_score": round(float(js[i]), 3),
                    "dup_of": items[dups[i]].id if i in dups else None,
                }
            )
    with open(art("clean_flags.jsonl"), "w") as f:
        for r in flagged:
            f.write(json.dumps(r) + "\n")

    junk_rows = [r["row"] for r in sorted(flagged, key=lambda r: -r["junk_score"]) if r["text_junk"]]
    dup_rows = [r["row"] for r in flagged if r["dup_of"] is not None]
    n_groups = len({dups[r] for r in dup_rows})
    print(f"text_junk: {len(junk_rows)}  near-dups: {len(dup_rows)} rows in {n_groups} groups "
          f"(threshold {args.dup_threshold})")
    if junk_rows:
        image_grid([items[r].path for r in junk_rows[:96]], art("clean_junk_grid.png"), cols=8,
                   labels=[f"{js[r]:.2f}" for r in junk_rows[:96]])
    # borderline cases just below threshold, to tune it
    jset = set(junk_rows)
    border = [int(i) for i in np.argsort(-js) if int(i) not in jset][:24]
    image_grid([items[r].path for r in border], art("clean_junk_borderline.png"), cols=8,
               labels=[f"{js[r]:.2f}" for r in border])
    pairs, seen = [], set()
    for r in dup_rows:  # one (representative, duplicate) pair per group
        if dups[r] in seen:
            continue
        seen.add(dups[r])
        pairs += [items[dups[r]].path, items[r].path]
        if len(pairs) >= 48:
            break
    if pairs:
        image_grid(pairs, art("clean_dup_grid.png"), cols=8)


if __name__ == "__main__":
    main()
