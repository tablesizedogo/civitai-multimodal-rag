"""Hugging Face Space entry point (built by prepare_space.py).

Data layout inside the Space: data/meta.jsonl (slim metadata with Civitai CDN
URLs) and artifacts/*.npy (precomputed embeddings). No images are stored; the
gallery loads them from image.civitai.com.

Set ENABLE_LLM=0 in the Space settings to hide the prompt-assistant tab (the
free CPU tier has 2 vCPU, so one answer takes roughly 40-70 s).
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("CIVITAI_DATA_DIR", str(ROOT / "data"))
os.environ.setdefault("CIVITAI_ART_DIR", str(ROOT / "artifacts"))

from civitai_search.app import CSS, build_demo  # noqa: E402

demo = build_demo(with_llm=os.environ.get("ENABLE_LLM", "1") == "1")

if __name__ == "__main__":
    demo.launch(css=CSS)
