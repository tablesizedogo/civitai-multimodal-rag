"""Index sizes and image->image query latency (text latency: eval_text2img).

    python -m civitai_search.bench
"""
from __future__ import annotations

import json
import random
import time

import numpy as np
from PIL import Image

from .config import art
from .data import load_items
from .retrieval import build_all

# Embedding build throughput recorded from the build logs (artifacts/*.log), CPU 8 vCPU.
BUILD_TIMES = {
    "clip_image": {"items": 30000, "seconds": 342, "per_s": 87.8, "log": "artifacts/clip.log"},
    "mclip_image": {"items": 30000, "seconds": 309, "per_s": 97.0, "log": "artifacts/mclip.log"},
    "clip_image_incremental_100k": {"items": 18809, "seconds": 255, "per_s": 73.8, "log": "artifacts/emb_final.log"},
    "mclip_image_incremental_100k": {"items": 41729, "seconds": 484, "per_s": 86.2, "log": "artifacts/emb_final.log"},
    "text_e5_incremental_100k": {"items": 70294, "seconds": 864, "per_s": 81.4, "log": "artifacts/emb_final.log"},
}


def _lat(fn, inputs) -> dict:
    fn(inputs[0])
    t = []
    for x in inputs:
        t0 = time.perf_counter()
        fn(x)
        t.append(time.perf_counter() - t0)
    return {"p50_ms": round(1000 * float(np.percentile(t, 50)), 1),
            "p95_ms": round(1000 * float(np.percentile(t, 95)), 1), "n": len(t)}


def main() -> None:
    rets = build_all()
    items = load_items()
    a = rets["A"]
    mb = 1 / 2**20
    sizes = {
        "corpus_rows_total": len(items),
        "corpus_rows_indexed": int(len(a.rows)),
        "A_bm25_vocab": len(a.vocab),
        "A_bm25_nnz": int(a.mat.nnz),
        "A_bm25_MB": round((a.mat.data.nbytes + a.mat.indices.nbytes + a.mat.indptr.nbytes) * mb, 1),
    }
    for key, npy in (("B", "text_e5.npy"), ("C", "clip_image.npy"), ("E", "mclip_image.npy")):
        r = rets[key]
        ix = r.index.index
        sizes[f"{key}_dim"] = ix.d
        sizes[f"{key}_faiss_MB_fp32"] = round(ix.ntotal * ix.d * 4 * mb, 1)
        sizes[f"{key}_npy_MB_fp16"] = round(art(npy).stat().st_size * mb, 1)
    paths = [items[i].path for i in random.Random(0).sample(range(len(items)), 50)]
    imgs = [Image.open(p).convert("RGB") for p in paths]
    lat = {"image_query_C": _lat(lambda im: rets["C"].search_image(im, 10), imgs),
           "image_query_E": _lat(lambda im: rets["E"].search_image(im, 10), imgs)}
    out = {"index_sizes": sizes, "image_query_latency": lat, "embedding_build": BUILD_TIMES}
    json.dump(out, open(art("eval", "bench_100k.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
