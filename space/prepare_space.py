"""Build a Hugging Face Space bundle (NOT deployed automatically).

    cd search && PYTHONPATH=. .venv/bin/python ../space/prepare_space.py [--out ../space/build] [--no-e5]

The Space holds only embeddings + slim metadata; images are shown from the
Civitai CDN (the `download_url` of each record), so no image files are
re-hosted. Output layout:

    build/
      app.py  requirements.txt  README.md      (copied from space/)
      civitai_search/                          (copy of the package)
      data/meta.jsonl                          (indexed rows only, slim fields)
      artifacts/{clip_image,mclip_image,text_e5}.npy, ids.txt, clean_flags.jsonl

Rows flagged as junk / near-duplicates are dropped from the bundle, so the
embedding matrices are re-sliced to stay aligned with load_items() order.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from civitai_search.config import META_PATH, PKG_ROOT, art
from civitai_search.data import load_items
from civitai_search.retrieval import load_keep_mask

SPACE = Path(__file__).resolve().parent
SLIM = ["id", "download_url", "page_url", "username", "baseModel", "prompt", "negativePrompt", "Model", "sampler",
        "scheduler", "steps", "cfgScale", "seed", "loras", "has_comfy_workflow", "nsfwLevel", "browsingLevel", "nsfw",
        "width", "height", "createdAt"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(SPACE / "build"))
    ap.add_argument("--no-e5", action="store_true", help="skip B/D/F (saves ~70 MB)")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "data").mkdir(parents=True, exist_ok=True)
    (out / "artifacts").mkdir(exist_ok=True)

    items = load_items()
    keep = load_keep_mask()
    keep_ids = {items[i].id for i in keep}
    rows = {}
    for line in open(META_PATH, encoding="utf-8"):
        r = json.loads(line)
        if r["id"] in keep_ids:
            s = {k: r.get(k) for k in SLIM}
            s["local_path"] = f"images/{r['id']}.webp"  # absent in the Space -> CDN URL is used
            rows[r["id"]] = s
    ordered = sorted(rows)  # same order as load_items() in the Space
    with open(out / "data" / "meta.jsonl", "w", encoding="utf-8") as f:
        for i in ordered:
            f.write(json.dumps(rows[i], ensure_ascii=False) + "\n")
    row_of = {it.id: n for n, it in enumerate(items)}
    sel = np.array([row_of[i] for i in ordered])
    names = ["clip_image.npy", "mclip_image.npy"] + ([] if args.no_e5 else ["text_e5.npy"])
    for name in names:
        np.save(out / "artifacts" / name, np.load(art(name))[sel].astype(np.float16))
    (out / "artifacts" / "ids.txt").write_text("\n".join(map(str, ordered)))
    (out / "artifacts" / "clean_flags.jsonl").write_text("")  # already filtered
    for f in ("app.py", "requirements.txt", "README.md"):
        shutil.copy(SPACE / f, out / f)
    dst = out / "civitai_search"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(PKG_ROOT / "civitai_search", dst, ignore=shutil.ignore_patterns("__pycache__"))
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 2**20
    print(f"bundle: {len(ordered)} rows, {size:.0f} MB -> {out}")


if __name__ == "__main__":
    main()
