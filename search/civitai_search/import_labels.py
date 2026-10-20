"""Turn eval/prelim_labels_raw.txt (contact-sheet indices) into eval/qrels_prelim.csv.

    python -m civitai_search.import_labels

Raw format, one line per query: "<qid>: <indices of relevant images>", ranges
like "7-29" allowed; indices refer to artifacts/eval/label_sheets/index.json.
Rows already present in qrels_prelim.csv (labels reused from the 30k run) are kept.
"""
from __future__ import annotations

import json
import re

from .config import PKG_ROOT, art
from .eval_text2img import PRELIM, load_queries, read_qrels, write_qrels

RAW = PKG_ROOT / "eval" / "prelim_labels_raw.txt"


def parse_raw(path=RAW) -> dict[int, set[int]]:
    out = {}
    for line in open(path, encoding="utf-8"):
        line = line.split("#")[0].strip()
        m = re.match(r"^(\d+):(.*)$", line)
        if not m:
            continue
        idx = set()
        for tok in m.group(2).split():
            if "-" in tok:
                a, b = map(int, tok.split("-"))
                idx.update(range(a, b + 1))
            else:
                idx.add(int(tok))
        out[int(m.group(1))] = idx
    return out


def main() -> None:
    index = {int(k): v for k, v in json.load(open(art("eval", "label_sheets", "index.json"))).items()}
    raw = parse_raw()
    queries = {q["qid"]: q for q in load_queries()}
    old = read_qrels(PRELIM)
    rows = [{"qid": q, "lang": queries[q]["lang"], "query": queries[q]["q"], "image_id": i, "relevant": v,
             "labeler": "assistant_30k_reused"} for (q, i), v in old.items()]
    missing = [q for q in index if q not in raw]
    if missing:
        raise SystemExit(f"no raw labels for queries {missing}")
    for qid, ids in index.items():
        bad = [n for n in raw[qid] if n >= len(ids)]
        if bad:
            raise SystemExit(f"q{qid}: indices {bad} out of range ({len(ids)} images)")
        for n, image_id in enumerate(ids):
            if (qid, image_id) in old:
                continue
            rows.append({"qid": qid, "lang": queries[qid]["lang"], "query": queries[qid]["q"], "image_id": image_id,
                         "relevant": int(n in raw[qid]), "labeler": "assistant_100k"})
    rows.sort(key=lambda r: (r["qid"], r["image_id"]))
    write_qrels(PRELIM, rows)
    print(f"{len(rows)} labels -> {PRELIM} ({sum(r['relevant'] for r in rows)} relevant)")


if __name__ == "__main__":
    main()
