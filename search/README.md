# `search/` — retrieval, RAG prompt assistant and evaluation

This folder contains the Python package `civitai_search` (indexing, retrieval methods A–F, the RAG prompt
assistant, the Gradio app and all evaluation scripts), the evaluation query sets and labels (`eval/`), and the
metric outputs (`artifacts/eval/`).

See the [top-level README](../README.md) for the architecture, results, how to run and limitations.

Quick start with the prebuilt index (from this folder, no collection needed):

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
python -m civitai_search.fetch_prebuilt      # ~400 MB from huggingface.co/datasets/AER6/civitai-multimodal-rag-data
python -m civitai_search.app --port 7860     # thumbnails load from the Civitai CDN
```

With a self-collected corpus (built by `../collect.py`), set `CIVITAI_DATA_DIR=..` (folder with `meta.jsonl` +
`images/`) instead of fetching the bundle; see the top-level README for the full pipeline.
