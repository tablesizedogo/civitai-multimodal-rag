"""Dataset statistics for the README.

    python -m civitai_search.dataset_stats
"""
from __future__ import annotations

import json
from collections import Counter

import numpy as np

from .config import META_PATH, art
from .data import load_items, model_family
from .textnorm import clean_prompt


def main() -> None:
    raw = [json.loads(l) for l in open(META_PATH, encoding="utf-8") if l.strip()]
    items = load_items()
    n = len(items)
    fam = Counter(model_family(it.base_model) for it in items)
    base = Counter(it.base_model for it in items)
    users = Counter(r.get("username") for r in raw)
    words = np.array([len(clean_prompt(it.prompt).split()) for it in items])
    created = sorted(r["createdAt"][:10] for r in raw if r.get("createdAt"))
    disk = sum(it.path.stat().st_size for it in items if it.path.exists())

    def share(pred):
        return round(sum(1 for it in items if pred(it)) / n, 3)

    flags = Counter()
    fp = art("clean_flags.jsonl")
    if fp.exists():
        for l in open(fp):
            r = json.loads(l)
            flags["text_junk"] += bool(r.get("text_junk"))
            flags["near_duplicate"] += r.get("dup_of") is not None
    out = {
        "images": n,
        "authors": len(users),
        "max_images_per_author": users.most_common(1)[0][1],
        "created_range": [created[0], created[-1]],
        "images_MB_on_disk": round(disk / 2**20),
        "family_counts": dict(fam.most_common()),
        "base_model_counts": dict(base.most_common()),
        "prompt_words_cleaned": {"median": int(np.median(words)), "p10": int(np.percentile(words, 10)),
                                 "p90": int(np.percentile(words, 90))},
        "metadata_completeness": {
            "prompt": share(lambda it: bool(it.prompt.strip())),
            "negative_prompt": share(lambda it: bool((it.negative_prompt or "").strip())),
            "checkpoint_name": share(lambda it: bool(it.model)),
            "sampler": share(lambda it: bool(it.sampler) and it.sampler.lower() != "undefined"),
            "steps": share(lambda it: it.steps is not None),
            "cfg": share(lambda it: it.cfg_scale is not None),
            "seed": share(lambda it: it.seed is not None),
            "loras": share(lambda it: bool(it.loras)),
            "comfy_workflow": share(lambda it: it.has_comfy_workflow),
        },
        "rating_filter": dict(Counter((r.get("nsfwLevel"), r.get("browsingLevel"), r.get("nsfw")) for r in raw).most_common()),
        "cleaning_flags": dict(flags),
    }
    out["rating_filter"] = {str(k): v for k, v in out["rating_filter"].items()}
    json.dump(out, open(art("eval", "dataset_stats.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
