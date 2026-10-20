"""Gradio demo with two tabs.

* Search: text (EN/RU) or image query -> gallery with generation metadata.
* Prompt assistant (RAG): idea -> retrieved references -> local LLM writes a
  grounded prompt + settings with citations (see generate.py).

    python -m civitai_search.app [--port 7860] [--share] [--no-llm]
"""
from __future__ import annotations

import argparse
import html

import gradio as gr

from .data import load_items
from .retrieval import build_all
from .safety import blocked_term

# Order = preference by the manual P@10 check (see README): E for mixed RU/EN,
# C for English-only.
METHODS = {
    "E: multilingual CLIP on images (EN/RU)": "E",
    "C: CLIP on images (EN)": "C",
    "D: hybrid e5 + CLIP": "D",
    "F: hybrid e5 + mCLIP": "F",
    "B: e5 on prompts": "B",
    "A: BM25 on prompts": "A",
}
GEN_METHODS = {k: v for k, v in METHODS.items() if v in ("E", "F", "C")}


def _caption(it) -> str:
    return f"#{it.id} | {it.base_model} | {it.sampler or '-'}"


def _details(hits, items) -> str:
    parts = []
    for rank, (row, score) in enumerate(hits, 1):
        it = items[row]
        loras = ", ".join(f"{l.get('name')}:{l.get('weight')}" if isinstance(l, dict) else str(l) for l in it.loras) or "-"
        parts.append(
            f"<details><summary><b>#{rank}</b> id {it.id} · {html.escape(it.base_model)} · score {score:.3f} · "
            f"<a href='{it.page_url}' target='_blank'>civitai</a></summary>"
            f"<p><b>Prompt:</b> {html.escape(it.prompt[:1500])}</p>"
            f"<p><b>Negative:</b> {html.escape((it.negative_prompt or '-')[:500])}</p>"
            f"<p><b>Model:</b> {html.escape(str(it.model or '-'))} · <b>sampler</b> {it.sampler} · "
            f"<b>steps</b> {it.steps} · <b>cfg</b> {it.cfg_scale} · <b>seed</b> {it.seed} · "
            f"<b>LoRA:</b> {html.escape(loras)} · <b>ComfyUI workflow:</b> {'yes' if it.has_comfy_workflow else 'no'}</p>"
            "</details>"
        )
    return "\n".join(parts)


# Wrap long prompts in the assistant answer instead of scrolling horizontally.
CSS = """
.gen-answer pre, .gen-answer code { white-space: pre-wrap !important; word-break: break-word; }
.gen-answer table { width: 100%; }
"""


def build_demo(with_llm: bool = True) -> gr.Blocks:
    items = load_items()
    rets = build_all()
    methods = [m for m, key in METHODS.items() if key in rets]
    gen_methods = [m for m, key in GEN_METHODS.items() if key in rets]
    assistant = {}

    def search(text, image, method, k):
        k = int(k)
        if image is not None:
            hits = rets["C"].search_image(image, k)  # image query: only CLIP applies
        elif text and text.strip():
            term = blocked_term(text)
            if term:
                return [], f"<p><b>Refused:</b> the query contains a blocked term ('{html.escape(term)}'). This demo is strictly SFW.</p>"
            hits = rets[METHODS[method]].search(text.strip(), k)
        else:
            return [], "Enter a query or upload an image."
        gallery = [(items[r].image_ref(), _caption(items[r])) for r, _ in hits]
        return gallery, _details(hits, items)

    def assist(idea, method, k):
        from .generate import PromptAssistant, render_markdown

        if not idea or not idea.strip():
            return [], "Describe your idea first.", ""
        if "pa" not in assistant:
            assistant["pa"] = PromptAssistant(rets, default_method=GEN_METHODS[method])
        ans = assistant["pa"].answer(idea.strip(), GEN_METHODS[method], int(k))
        gallery = [(items[r].image_ref(), _caption(items[r])) for r, _ in ans.hits]
        return gallery, render_markdown(ans), _details(ans.hits, items)

    with gr.Blocks(title="Civitai prompt search + RAG assistant") as demo:
        gr.Markdown("## Civitai prompt search & prompt assistant\n"
                    "100k strictly-SFW AI images from Civitai with their prompts and generation settings. "
                    "Queries in English or Russian.")
        with gr.Tab("Search"):
            with gr.Row():
                with gr.Column(scale=1):
                    text = gr.Textbox(label="Text query", placeholder="foggy forest at dawn / лиса в осеннем лесу")
                    image = gr.Image(label="...or an image (find similar)", type="pil")
                    method = gr.Radio(methods, value=methods[0], label="Method (text queries)")
                    k = gr.Slider(4, 40, value=12, step=4, label="Number of results")
                    btn = gr.Button("Search", variant="primary")
                with gr.Column(scale=3):
                    gallery = gr.Gallery(label="Results", columns=4, height=640)
                    details = gr.HTML()
            btn.click(search, [text, image, method, k], [gallery, details])
            text.submit(search, [text, image, method, k], [gallery, details])
        if with_llm:
            with gr.Tab("Prompt assistant (RAG)"):
                gr.Markdown("Describe what you want to generate. The assistant retrieves similar Civitai images and a "
                            "small local LLM (Qwen2.5-1.5B-Instruct, CPU) writes a prompt and settings **copied from the "
                            "retrieved images**, with citations. Takes ~10–25 s on CPU; the first call also loads the model.")
                with gr.Row():
                    with gr.Column(scale=1):
                        idea = gr.Textbox(label="Your idea", lines=3,
                                          placeholder="a cozy cabin in a snowy forest at night / киберпанк-город под дождём, аниме")
                        gmethod = gr.Radio(gen_methods, value=gen_methods[0], label="Retriever")
                        gk = gr.Slider(3, 10, value=6, step=1, label="References given to the LLM")
                        gbtn = gr.Button("Write prompt", variant="primary")
                    with gr.Column(scale=2):
                        ggallery = gr.Gallery(label="References (retrieved)", columns=3, height=420)
                with gr.Row():
                    with gr.Column(scale=3):
                        answer = gr.Markdown(elem_classes=["gen-answer"])
                    with gr.Column(scale=2):
                        gdetails = gr.HTML()
                gbtn.click(assist, [idea, gmethod, gk], [ggallery, answer, gdetails])
                idea.submit(assist, [idea, gmethod, gk], [ggallery, answer, gdetails])
    return demo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--no-llm", action="store_true", help="search tab only")
    args = ap.parse_args()
    from .config import DATA_DIR

    build_demo(not args.no_llm).launch(server_port=args.port, share=args.share, allowed_paths=[str(DATA_DIR)],
                                        css=CSS)


if __name__ == "__main__":
    main()
