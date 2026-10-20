"""Evaluation of the generation step (prompt assistant).

    python -m civitai_search.eval_generation [--modes constrained,free] [--gguf PATH] [--tag 1.5b]

For each idea in eval/generation_queries.json (13 EN + 13 RU) the assistant
retrieves k=6 references with method E and the LLM answers. We record:

* json_ok             - the LLM output parsed as JSON
* grounded@raw        - share of recommended items (base model, checkpoint,
                        sampler, steps, CFG, each LoRA) whose RAW LLM value
                        occurs in some retrieved reference
* coherent@raw        - ...occurs in a reference with the recommended base model
                        (so steps/CFG/LoRAs belong to a consistent recipe)
* grounded@final      - after the post-check (should be 1.0 by construction)
* corrected           - items the post-check had to replace
* citation_valid      - share of cited source ids that are among the references
* clip_prompt_query   - mCLIP text-text cosine(generated prompt, idea)
* clip_prompt_refs    - mean mCLIP cosine(generated prompt, reference images)
  baselines: the top-1 reference's own prompt vs idea, and the idea vs references
* latency             - retrieval + LLM seconds
* refusal probes      - NSFW ideas that must be refused before retrieval

Outputs (artifacts/eval/): generation_outputs_<tag>_<mode>.jsonl,
generation_review_<tag>.md (side-by-side for manual reading),
generation_rating_<tag>.csv (empty columns for a 1-5 manual rating),
generation_metrics_<tag>.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import time

import numpy as np

from .config import PKG_ROOT, art
from .data import load_items

QUERIES = PKG_ROOT / "eval" / "generation_queries.json"


def _cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def item_scores(ans) -> dict:
    s = ans.settings
    fields = [s[f] for f in ("base_model", "checkpoint", "sampler", "steps", "cfg") if f in s]
    n_items = len(fields) + len(ans.loras) + len([d for d in ans.dropped_loras if d["reason"] != "unsafe name"])
    g_raw = sum(f.grounded for f in fields) + len(ans.loras) + sum(bool(d.get("grounded")) for d in ans.dropped_loras)
    c_raw = sum(f.coherent for f in fields) + len(ans.loras)
    final_ok = sum(bool(f.support) or f.value is None for f in fields) + sum(bool(l["support"]) for l in ans.loras)
    n_final = len(fields) + len(ans.loras)
    n_src = len(ans.sources) + len(ans.invalid_sources)
    return {
        "n_items": n_items,
        "grounded_raw": g_raw / n_items if n_items else 1.0,
        "coherent_raw": c_raw / n_items if n_items else 1.0,
        "grounded_final": final_ok / n_final if n_final else 1.0,
        "n_corrected": sum(f.corrected for f in fields) + len(ans.dropped_loras),
        "citation_valid": len(ans.sources) / n_src if n_src else None,
        "n_loras_raw": len(ans.loras) + len(ans.dropped_loras),
    }


def run(modes: list[str], gguf: str | None, tag: str, k: int = 6) -> dict:
    from .encoders import MCLIP_MODEL, MCLIP_PRETRAINED, get_clip
    from .generate import PromptAssistant, get_llm, render_markdown
    from .retrieval import build_all

    cfg = json.load(open(QUERIES, encoding="utf-8"))
    items = load_items()
    rets = build_all(only="E")
    llm = get_llm(gguf)
    pa = PromptAssistant(rets, llm, default_method="E")
    mclip = get_clip(MCLIP_MODEL, MCLIP_PRETRAINED)
    img_emb = rets["E"].emb
    pa.answer("warm up", k=2)  # first call pays model warm-up

    metrics = {"llm": llm.name, "retriever": "E (mCLIP)", "k": k, "modes": {}}
    review = {}
    for mode in modes:
        outs, rows = [], []
        for q in cfg["ideas"]:
            ans = pa.answer(q["q"], "E", k, mode)
            rec = {"query": q["q"], "lang": q["lang"], "mode": mode, **ans.to_dict(), "markdown": render_markdown(ans)}
            if not ans.refused:
                sc = item_scores(ans)
                ref_rows = [r for r, _ in ans.hits[:k]]
                e_prompt, e_query, e_top1 = mclip.encode_texts([ans.prompt, q["q"], items[ref_rows[0]].prompt[:300]])
                ref = img_emb[ref_rows].astype(np.float32)
                sc.update({
                    "clip_prompt_query": _cos(e_prompt, e_query),
                    "clip_prompt_refs": float(np.mean([_cos(e_prompt, v) for v in ref])),
                    "clip_top1prompt_query": _cos(e_top1, e_query),
                    "clip_query_refs": float(np.mean([_cos(e_query, v) for v in ref])),
                    "llm_s": ans.timings["llm_s"], "total_s": ans.timings["total_s"],
                    "completion_tokens": ans.timings["completion_tokens"], "prompt_tokens": ans.timings["prompt_tokens"],
                    "sfw_removed": len(ans.removed_terms), "json_ok": ans.json_ok,
                })
                rec["scores"] = sc
                rows.append(sc)
            outs.append(rec)
            print(mode, q["q"][:40], rec.get("scores", {}).get("grounded_raw"), ans.timings.get("total_s"), flush=True)
        with open(art("eval", f"generation_outputs_{tag}_{mode}.jsonl"), "w", encoding="utf-8") as f:
            for r in outs:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        review[mode] = outs

        def avg(key):
            v = [r[key] for r in rows if r.get(key) is not None]
            return round(float(np.mean(v)), 3) if v else None

        tot = [r["total_s"] for r in rows]
        metrics["modes"][mode] = {
            "n": len(cfg["ideas"]), "answered": len(rows),
            "json_ok": avg("json_ok"),
            "grounded_raw": avg("grounded_raw"), "coherent_raw": avg("coherent_raw"),
            "grounded_final": avg("grounded_final"),
            "corrected_items_per_answer": avg("n_corrected"),
            "citation_valid": avg("citation_valid"),
            "clip_prompt_query": avg("clip_prompt_query"), "clip_top1prompt_query": avg("clip_top1prompt_query"),
            "clip_prompt_refs": avg("clip_prompt_refs"), "clip_query_refs": avg("clip_query_refs"),
            "latency_total_p50_s": round(float(np.percentile(tot, 50)), 1),
            "latency_total_p95_s": round(float(np.percentile(tot, 95)), 1),
            "completion_tokens_mean": avg("completion_tokens"), "prompt_tokens_mean": avg("prompt_tokens"),
            "sfw_terms_removed": int(sum(r["sfw_removed"] for r in rows)),
        }
        print(json.dumps(metrics["modes"][mode], indent=1))

    probes = [pa.answer(p, "E", k) for p in cfg["refusal_probes"]]
    metrics["refusal_probes"] = {"n": len(probes), "refused": sum(bool(a.refused) for a in probes),
                                 "examples": [{"q": a.query, "refused": a.refused} for a in probes]}
    json.dump(metrics, open(art("eval", f"generation_metrics_{tag}.json"), "w"), indent=1, ensure_ascii=False)

    # human-readable review + rating sheet
    main_mode = modes[0]
    with open(art("eval", f"generation_review_{tag}.md"), "w", encoding="utf-8") as f:
        f.write(f"# Generation outputs for manual review ({llm.name}, retriever E, k={k})\n\n")
        for i, rec in enumerate(review[main_mode]):
            f.write(f"## {i}. {rec['query']}\n\nReferences: " +
                    ", ".join(f"[#{x}](https://civitai.com/images/{x})" for x in rec["ref_ids"]) + "\n\n")
            for mode in modes:
                f.write(f"**mode: {mode}**\n\n{review[mode][i]['markdown']}\n\n")
    with open(art("eval", f"generation_rating_{tag}.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx", "query", "mode", "prompt", "base_model", "sampler", "steps", "cfg",
                    "rating_relevance_1to5", "rating_prompt_quality_1to5", "rating_settings_sensible_1to5", "notes"])
        for mode in modes:
            for i, rec in enumerate(review[mode]):
                s = rec.get("settings") or {}
                g = lambda k: (s.get(k) or {}).get("value")
                w.writerow([i, rec["query"], mode, rec.get("prompt"), g("base_model"), g("sampler"), g("steps"), g("cfg"),
                            "", "", "", ""])
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="constrained,free")
    ap.add_argument("--gguf", default=None)
    ap.add_argument("--tag", default="qwen2.5-1.5b")
    ap.add_argument("--k", type=int, default=6)
    args = ap.parse_args()
    run(args.modes.split(","), args.gguf, args.tag, args.k)


if __name__ == "__main__":
    main()
