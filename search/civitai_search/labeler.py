"""Tiny Gradio labeler for text->image relevance (to replace the preliminary labels).

    python -m civitai_search.labeler [--port 7862] [--show-prelim]

Pick a query, tick the images that are relevant (main subject + most
attributes visible), press Save. Labels go to eval/qrels_user.csv (all shown
images are written: ticked = 1, others = 0). Then:

    python -m civitai_search.eval_text2img score --qrels eval/qrels_user.csv
    python -m civitai_search.eval_text2img agree    # Cohen's kappa vs preliminary labels

By default the preliminary labels are hidden to avoid anchoring.
Labeling ~20 queries (about 600 images) takes roughly 20-30 minutes.
"""
from __future__ import annotations

import argparse

import gradio as gr

from .eval_text2img import PRELIM, USER, load_queries, pooled, read_qrels, write_qrels


def build(show_prelim: bool = False) -> gr.Blocks:
    pool = pooled()
    queries = {q["qid"]: q for q in load_queries()}
    by_q: dict[int, list[dict]] = {}
    for r in pool:
        by_q.setdefault(r["qid"], []).append(r)
    prelim = read_qrels(PRELIM)
    choices = [f"{q}: [{queries[q]['lang']}] {queries[q]['q']}" for q in sorted(by_q)]

    def status():
        user = read_qrels(USER)
        done = sorted({q for q, _ in user})
        return f"Labeled queries: {len(done)} / {len(by_q)}" + (f" (last: {done[-1]})" if done else "")

    def show(choice):
        qid = int(choice.split(":")[0])
        rs = by_q[qid]
        user = read_qrels(USER)
        gallery = [(r["local_path"], str(n)) for n, r in enumerate(rs)]
        nums = [str(n) for n in range(len(rs))]
        if any((qid, r["image_id"]) in user for r in rs):
            ticked = [str(n) for n, r in enumerate(rs) if user.get((qid, r["image_id"]))]
        elif show_prelim:
            ticked = [str(n) for n, r in enumerate(rs) if prelim.get((qid, r["image_id"]))]
        else:
            ticked = []
        return gallery, gr.CheckboxGroup(choices=nums, value=ticked), status()

    def save(choice, ticked):
        qid = int(choice.split(":")[0])
        rs = by_q[qid]
        user = read_qrels(USER)
        rows = [{"qid": q, "lang": queries[q]["lang"], "query": queries[q]["q"], "image_id": i, "relevant": v,
                 "labeler": "user"} for (q, i), v in user.items() if q != qid]
        tick = {int(t) for t in ticked or []}
        rows += [{"qid": qid, "lang": r["lang"], "query": r["query"], "image_id": r["image_id"],
                  "relevant": int(n in tick), "labeler": "user"} for n, r in enumerate(rs)]
        rows.sort(key=lambda r: (r["qid"], r["image_id"]))
        write_qrels(USER, rows)
        nxt = choices[min(choices.index(choice) + 1, len(choices) - 1)]
        return status(), nxt

    with gr.Blocks(title="Relevance labeler") as demo:
        gr.Markdown("### Text→image relevance labeler\nRelevant = the main subject of the query and most of its "
                    "attributes are visible in the image. Images are pooled from the top-10 of all methods A–F, "
                    "shown in a fixed order (method names hidden).")
        with gr.Row():
            q = gr.Dropdown(choices, value=choices[0], label="Query")
            st = gr.Markdown(status())
        gal = gr.Gallery(columns=8, height=620, label="Pooled images (number = caption)")
        ticks = gr.CheckboxGroup([str(n) for n in range(max(len(v) for v in by_q.values()))], label="Relevant images")
        save_btn = gr.Button("Save & next", variant="primary")
        q.change(show, q, [gal, ticks, st])
        save_btn.click(save, [q, ticks], [st, q])
        demo.load(show, q, [gal, ticks, st])
    return demo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7862)
    ap.add_argument("--show-prelim", action="store_true", help="pre-tick the preliminary labels")
    args = ap.parse_args()
    from .config import DATA_DIR

    build(args.show_prelim).launch(server_port=args.port, allowed_paths=[str(DATA_DIR)])


if __name__ == "__main__":
    main()
