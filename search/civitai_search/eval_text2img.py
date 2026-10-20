"""Text->image evaluation of methods A-F on a fixed query set (100k corpus).

    python -m civitai_search.eval_text2img run      # run queries, save top-10, latency, pooled label sheet
    python -m civitai_search.eval_text2img pool     # contact sheets of unlabeled pooled images (for labeling)
    python -m civitai_search.eval_text2img score    # P@10 / nDCG@10 / MRR@10 from eval/qrels_prelim.csv
    python -m civitai_search.eval_text2img score --qrels eval/qrels_user.csv   # your own labels
    python -m civitai_search.eval_text2img agree    # agreement between prelim and user labels

Relevance is binary and judged on the *image*: 1 if the main subject of the
query and most of its attributes are visible. Labels are pooled: every image
any method returned in its top 10 for a query is judged once, so all methods
are scored against the same pool (nDCG ideal = min(10, relevant in pool)).

The preliminary labels in eval/qrels_prelim.csv were made by an AI assistant
looking at contact sheets, not by a human annotator. Treat the numbers as an
estimate. Use `python -m civitai_search.labeler` to relabel a subset yourself.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np

from .config import PKG_ROOT, art
from .data import load_items

QUERIES = PKG_ROOT / "eval" / "queries_100k.json"
PRELIM = PKG_ROOT / "eval" / "qrels_prelim.csv"
USER = PKG_ROOT / "eval" / "qrels_user.csv"
RESULTS = "text2img_results_100k.json"
METHODS = "ABCDEF"
QRELS_FIELDS = ["qid", "lang", "query", "image_id", "relevant", "labeler"]


def load_queries() -> list[dict]:
    return json.load(open(QUERIES, encoding="utf-8"))


def read_qrels(path: Path) -> dict[tuple[int, int], int]:
    if not Path(path).exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return {(int(r["qid"]), int(r["image_id"])): int(r["relevant"])
                for r in csv.DictReader(f) if r.get("relevant", "") != ""}


def write_qrels(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, QRELS_FIELDS)
        w.writeheader()
        w.writerows(rows)


def run(methods: str = METHODS) -> None:
    from .retrieval import build_all

    items = load_items()
    rets = build_all(only=methods)
    queries = load_queries()
    for r in rets.values():  # warm-up (model load / first-call overhead)
        r.search("warm up", 10)
    out, lat = [], {k: [] for k in rets}
    for q in queries:
        rec = {**q, "results": {}}
        for key, r in rets.items():
            t0 = time.perf_counter()
            hits = r.search(q["q"], 10)
            lat[key].append(time.perf_counter() - t0)
            rec["results"][key] = [items[h[0]].id for h in hits]
        out.append(rec)
    json.dump(out, open(art("eval", RESULTS), "w"), ensure_ascii=False, indent=1)
    lat_ms = {k: {"p50_ms": round(1000 * float(np.percentile(v, 50)), 1),
                  "p95_ms": round(1000 * float(np.percentile(v, 95)), 1),
                  "mean_ms": round(1000 * float(np.mean(v)), 1), "n": len(v)} for k, v in lat.items()}
    json.dump(lat_ms, open(art("eval", "text_latency_100k.json"), "w"), indent=1)
    print(json.dumps(lat_ms, indent=1))
    export_sheet()


def pooled() -> list[dict]:
    res = json.load(open(art("eval", RESULTS), encoding="utf-8"))
    items = {it.id: it for it in load_items()}
    rows = []
    for r in res:
        seen = []
        for k in METHODS:
            for i in r["results"].get(k, []):
                if i not in seen:
                    seen.append(i)
        for i in seen:
            rows.append({"qid": r["qid"], "lang": r["lang"], "query": r["q"], "image_id": i,
                         "page_url": items[i].page_url, "local_path": str(items[i].path)})
    return rows


def export_sheet() -> Path:
    """CSV for labeling in a spreadsheet: one row per pooled (query, image)."""
    prelim = read_qrels(PRELIM)
    out = PKG_ROOT / "eval" / "labeling_sheet.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ["qid", "lang", "query", "image_id", "page_url", "local_path", "prelim", "your_label"])
        w.writeheader()
        for r in pooled():
            w.writerow({**r, "prelim": prelim.get((r["qid"], r["image_id"]), ""), "your_label": ""})
    print("labeling sheet:", out)
    return out


def contact_sheets(per_sheet: int = 2, only_unlabeled: bool = True) -> None:
    """Numbered grids of pooled images, `per_sheet` queries per PNG (for the labeler)."""
    from PIL import Image, ImageDraw

    from .viz import _font, _thumb

    prelim = read_qrels(PRELIM)
    rows = pooled()
    by_q: dict[int, list[dict]] = {}
    for r in rows:
        if only_unlabeled and (r["qid"], r["image_id"]) in prelim:
            continue
        by_q.setdefault(r["qid"], []).append(r)
    qids = sorted(by_q)
    tile, cols = 150, 10
    outdir = art("eval", "label_sheets", "x").parent
    index = {}
    for s in range(0, len(qids), per_sheet):
        blocks = []
        for qid in qids[s:s + per_sheet]:
            rs = by_q[qid]
            nrows = math.ceil(len(rs) / cols)
            g = Image.new("RGB", (cols * tile, 30 + nrows * tile), (20, 20, 20))
            d = ImageDraw.Draw(g)
            d.text((6, 5), f'q{qid}: {rs[0]["query"]}', fill=(255, 255, 255), font=_font(18))
            for n, r in enumerate(rs):
                x, y = (n % cols) * tile, 30 + (n // cols) * tile
                g.paste(_thumb(Path(r["local_path"]), tile), (x, y))
                d.rectangle([x, y, x + 26, y + 18], fill=(0, 0, 0))
                d.text((x + 3, y + 1), str(n), fill=(255, 255, 0), font=_font(14))
            blocks.append(g)
            index[qid] = [r["image_id"] for r in rs]
        h = sum(b.height for b in blocks) + 6 * len(blocks)
        sheet = Image.new("RGB", (cols * tile, h), (60, 60, 60))
        y = 0
        for b in blocks:
            sheet.paste(b, (0, y))
            y += b.height + 6
        sheet.save(outdir / f"sheet_{s // per_sheet:02d}.jpg", quality=88)
    json.dump(index, open(outdir / "index.json", "w"))
    print(f"{len(qids)} queries with unlabeled images -> {outdir}")


def _metrics(rels: list[int], n_rel_pool: int, k: int = 10) -> dict:
    rels = (rels + [0] * k)[:k]
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rels))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(k, n_rel_pool)))
    first = next((i for i, r in enumerate(rels) if r), None)
    return {"P@10": sum(rels) / k, "nDCG@10": dcg / ideal if ideal else 0.0,
            "MRR@10": 1 / (first + 1) if first is not None else 0.0}


def score(qrels_path: Path = PRELIM, tag: str = "prelim") -> dict:
    res = json.load(open(art("eval", RESULTS), encoding="utf-8"))
    qrels = read_qrels(qrels_path)
    judged_q = {q for q, _ in qrels}
    res = [r for r in res if r["qid"] in judged_q]
    table, unjudged = {}, {k: 0 for k in METHODS}
    for lang in ("en", "ru", "all"):
        sel = [r for r in res if lang == "all" or r["lang"] == lang]
        if not sel:
            continue
        table[lang] = {"n_queries": len(sel)}
        for k in METHODS:
            ms = []
            for r in sel:
                ids = r["results"].get(k, [])
                rels = [qrels.get((r["qid"], i), 0) for i in ids]
                if lang == "all":
                    unjudged[k] += sum((r["qid"], i) not in qrels for i in ids)
                n_rel = sum(v for (q, _), v in qrels.items() if q == r["qid"])
                ms.append(_metrics(rels, n_rel))
            table[lang][k] = {m: round(float(np.mean([x[m] for x in ms])), 3) for m in ms[0]}
    # parallel EN/RU pairs: same query in both languages
    queries = {q["qid"]: q for q in load_queries()}
    by_text = {q["q"]: q["qid"] for q in queries.values()}
    pairs = [(by_text[q["translation_of"]], q["qid"]) for q in queries.values()
             if q.get("translation_of") in by_text and q["qid"] in judged_q and by_text[q["translation_of"]] in judged_q]
    res_by = {r["qid"]: r for r in res}
    parallel = {}
    if pairs:
        for k in METHODS:
            p = {lang: float(np.mean([sum(qrels.get((qid, i), 0) for i in res_by[qid]["results"].get(k, [])[:10]) / 10
                                      for qid in ids]))
                 for lang, ids in (("en", [a for a, _ in pairs]), ("ru", [b for _, b in pairs]))}
            parallel[k] = {"P@10_en": round(p["en"], 3), "P@10_ru": round(p["ru"], 3)}
        print(f"\nparallel EN/RU pairs (n={len(pairs)}):", parallel)
    out = {"labels": str(qrels_path.name), "parallel_pairs": {"n": len(pairs), "P@10": parallel}, "note": "binary relevance, pooled top-10 of A-F; "
           + ("PRELIMINARY labels by the AI assistant" if tag == "prelim" else "user labels"),
           "metrics": table, "unjudged_in_top10": unjudged}
    json.dump(out, open(art("eval", f"text2img_metrics_{tag}.json"), "w"), indent=1, ensure_ascii=False)
    for lang, row in table.items():
        print(f"\n[{lang}] n={row['n_queries']}")
        print("metric   " + "  ".join(f"{k:>6}" for k in METHODS))
        for m in ("P@10", "nDCG@10", "MRR@10"):
            print(f"{m:8} " + "  ".join(f"{row[k][m]:6.3f}" for k in METHODS))
    print("unjudged in top-10:", unjudged)
    return out


def agree() -> None:
    a, b = read_qrels(PRELIM), read_qrels(USER)
    common = sorted(set(a) & set(b))
    if not common:
        print("no overlapping labels yet")
        return
    x = np.array([a[c] for c in common])
    y = np.array([b[c] for c in common])
    po = float((x == y).mean())
    pe = float(x.mean() * y.mean() + (1 - x.mean()) * (1 - y.mean()))
    kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
    out = {"n": len(common), "agreement": round(po, 3), "cohen_kappa": round(kappa, 3),
           "prelim_pos_rate": round(float(x.mean()), 3), "user_pos_rate": round(float(y.mean()), 3)}
    json.dump(out, open(art("eval", "label_agreement.json"), "w"), indent=1)
    print(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "pool", "score", "agree", "sheet"])
    ap.add_argument("--methods", default=METHODS)
    ap.add_argument("--qrels", default=None)
    args = ap.parse_args()
    if args.cmd == "run":
        run(args.methods)
    elif args.cmd == "pool":
        contact_sheets()
    elif args.cmd == "sheet":
        export_sheet()
    elif args.cmd == "score":
        path = Path(args.qrels) if args.qrels else PRELIM
        score(path, "prelim" if path == PRELIM else "user")
    else:
        agree()


if __name__ == "__main__":
    main()
