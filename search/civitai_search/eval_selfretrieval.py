"""Automatic prompt-based Recall@K (no labels needed).

    python -m civitai_search.eval_selfretrieval paraphrase [--n-llm 300]   # LLM paraphrases (cached)
    python -m civitai_search.eval_selfretrieval run                        # Recall@K for A-F

For a fixed random sample of indexed images (seed 0, n=1000, cleaned prompt
with >= 8 words) we turn the image's own prompt into a query and check whether
the image itself comes back in the top K. Using the full prompt would be
trivial for the prompt-based methods, so queries are degraded:

* trunc10      - the first 10 words of the cleaned prompt
* bag50        - a random 50% of the content words, shuffled (keyword query)
* paraphrase   - the local LLM rewrites the prompt as a short caption in other
                 words (<= 20 words); subset of --n-llm images

Two hit criteria: *strict* (exact image id) and *lenient* (any image whose
cleaned prompt is identical to the target's - authors often post series with
the same prompt, which are indistinguishable from the text side).

Caveat: this measures how well a method recovers an image from (a degraded
version of) its own prompt. It favours prompt-based methods (A, B), because the
query comes from the same text they index; CLIP methods must bridge
text->pixels. It complements, not replaces, the human-judged P@10.
"""
from __future__ import annotations

import argparse
import json
import random
import time

import numpy as np

from .config import art
from .data import load_items
from .textnorm import STOP, clean_prompt

N, SEED, KS = 1000, 0, (1, 5, 10, 100)
PARA = "selfret_paraphrases.jsonl"


def sample_rows() -> list[int]:
    from .retrieval import load_keep_mask

    items = load_items()
    keep = load_keep_mask()
    ok = [int(r) for r in keep if len(clean_prompt(items[r].prompt).split()) >= 8]
    return sorted(random.Random(SEED).sample(ok, N))


def make_queries(rows: list[int]) -> dict[str, dict[int, str]]:
    items = load_items()
    rng = random.Random(SEED + 1)
    q = {"trunc10": {}, "bag50": {}}
    for r in rows:
        words = clean_prompt(items[r].prompt).replace(",", " ").split()
        q["trunc10"][r] = " ".join(words[:10])
        content = [w for w in words if w.lower() not in STOP]
        k = max(3, len(content) // 2)
        bag = rng.sample(content, min(k, len(content)))
        rng.shuffle(bag)
        q["bag50"][r] = " ".join(bag)
    p = art("eval", PARA)
    if p.exists():
        q["paraphrase"] = {int(json.loads(l)["row"]): json.loads(l)["paraphrase"] for l in open(p, encoding="utf-8")}
    return q


def paraphrase(n_llm: int) -> None:
    from .generate import get_llm

    items = load_items()
    rows = sample_rows()[:]
    rows = sorted(random.Random(SEED + 2).sample(rows, n_llm))
    llm = get_llm().llm
    out = art("eval", PARA)
    done = set()
    if out.exists():
        done = {json.loads(l)["row"] for l in open(out, encoding="utf-8")}
    sys_p = ("Rewrite the image prompt as one short plain-English caption of at most 20 words, using different "
             "words where possible. Describe only what is depicted. Output only the caption.")
    t0 = time.time()
    with open(out, "a", encoding="utf-8") as f:
        for n, r in enumerate(rows):
            if r in done:
                continue
            text = clean_prompt(items[r].prompt)[:700]
            res = llm.create_chat_completion([{"role": "system", "content": sys_p}, {"role": "user", "content": text}],
                                             max_tokens=48, temperature=0.3, seed=0)
            cap = res["choices"][0]["message"]["content"].strip().strip('"').split("\n")[0]
            f.write(json.dumps({"row": r, "id": items[r].id, "paraphrase": cap}, ensure_ascii=False) + "\n")
            f.flush()
            if n % 25 == 0:
                print(n, round(time.time() - t0), cap[:90], flush=True)


def run(methods: str = "ABCDEF") -> None:
    from .retrieval import build_all

    items = load_items()
    rows = sample_rows()
    queries = make_queries(rows)
    norm = [None] * len(items)
    target_prompt = {r: clean_prompt(items[r].prompt).lower() for r in rows}
    rets = build_all(only=methods)
    res = {"n_sample": len(rows), "seed": SEED, "variants": {}}
    for var, qs in queries.items():
        res["variants"][var] = {"n": len(qs)}
        for key, r in rets.items():
            strict, lenient = {k: 0 for k in KS}, {k: 0 for k in KS}
            t0 = time.perf_counter()
            for row, text in qs.items():
                hits = [h[0] for h in r.search(text, max(KS))]
                rank = next((i for i, h in enumerate(hits) if h == row), None)
                lrank = None
                for i, h in enumerate(hits):
                    if h == row:
                        lrank = i
                        break
                    if norm[h] is None:
                        norm[h] = clean_prompt(items[h].prompt).lower()
                    if norm[h] == target_prompt[row]:
                        lrank = i
                        break
                for k in KS:
                    strict[k] += rank is not None and rank < k
                    lenient[k] += lrank is not None and lrank < k
            n = len(qs)
            res["variants"][var][key] = {
                **{f"R@{k}": round(strict[k] / n, 3) for k in KS},
                **{f"R@{k}_lenient": round(lenient[k] / n, 3) for k in KS},
                "sec": round(time.perf_counter() - t0, 1)}
            print(var, key, res["variants"][var][key], flush=True)
    examples = {var: [{"id": items[r].id, "query": qs[r]} for r in list(qs)[:5]] for var, qs in queries.items()}
    res["examples"] = examples
    json.dump(res, open(art("eval", "selfretrieval.json"), "w"), indent=1, ensure_ascii=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["paraphrase", "run"])
    ap.add_argument("--n-llm", type=int, default=300)
    ap.add_argument("--methods", default="ABCDEF")
    args = ap.parse_args()
    paraphrase(args.n_llm) if args.cmd == "paraphrase" else run(args.methods)


if __name__ == "__main__":
    main()
