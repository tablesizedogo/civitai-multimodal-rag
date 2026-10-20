"""Image->image retrieval under augmentations ("how was this picture made?").

For N random corpus images we build an augmented copy, query the CLIP index with
it and check where the original lands. Only method C applies here: there is no
text in the query. Writes artifacts/eval/img2img.json.

Caveat: the original itself is in the index, so this measures robustness of
near-copy retrieval (re-uploads, crops, screenshots), not semantic similarity.
"""
from __future__ import annotations

import argparse
import io
import json
import random
import time

import numpy as np
from PIL import Image, ImageEnhance

from .config import art
from .data import load_items
from .retrieval import ClipRetriever, load_keep_mask


def crop(im: Image.Image, rng: random.Random) -> Image.Image:
    s = rng.uniform(0.5, 0.8)  # keep 50-80% of each side (25-64% of area)
    w, h = im.size
    cw, ch = int(w * s), int(h * s)
    x, y = rng.randint(0, w - cw), rng.randint(0, h - ch)
    return im.crop((x, y, x + cw, y + ch))


def resize(im: Image.Image, rng: random.Random) -> Image.Image:
    w, h = im.size
    small = im.resize((max(32, w // 4), max(32, h // 4)), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


def color(im: Image.Image, rng: random.Random) -> Image.Image:
    im = ImageEnhance.Brightness(im).enhance(rng.uniform(0.6, 1.4))
    im = ImageEnhance.Contrast(im).enhance(rng.uniform(0.6, 1.4))
    im = ImageEnhance.Color(im).enhance(rng.uniform(0.3, 1.7))
    return im


def jpeg(im: Image.Image, rng: random.Random, q: int = 20) -> Image.Image:
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=q)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def combo(im: Image.Image, rng: random.Random) -> Image.Image:
    return jpeg(color(resize(crop(im, rng), rng), rng), rng, q=40)


AUGS = {"crop": crop, "resize_x4_down": resize, "color_jitter": color, "jpeg_q20": jpeg, "all_combined": combo}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    items = load_items()
    keep = load_keep_mask()
    c = ClipRetriever(keep)
    rng = random.Random(args.seed)
    sample = [int(x) for x in rng.sample(list(keep), args.n)]
    results, examples = {}, {}
    for name, fn in AUGS.items():
        ranks, t = [], []
        for row in sample:
            im = Image.open(items[row].path).convert("RGB")
            q = fn(im, random.Random(row))
            t0 = time.perf_counter()
            hits = c.search_image(q, k=10)
            t.append(time.perf_counter() - t0)
            got = [h[0] for h in hits]
            ranks.append(got.index(row) + 1 if row in got else 99)
        ranks = np.array(ranks)
        results[name] = {
            "R@1": float((ranks <= 1).mean()),
            "R@5": float((ranks <= 5).mean()),
            "R@10": float((ranks <= 10).mean()),
            "ms_per_query_incl_encode": 1000 * float(np.mean(t)),
        }
        print(name, results[name])
        examples[name] = sample[0]
        q = fn(Image.open(items[sample[0]].path).convert("RGB"), random.Random(sample[0]))
        q.save(art("eval", f"aug_example_{name}.jpg"))
    json.dump({"n": args.n, "seed": args.seed, "results": results}, open(art("eval", "img2img.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
