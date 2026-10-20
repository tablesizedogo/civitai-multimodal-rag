# `search/` — retrieval, RAG prompt assistant and evaluation

This folder contains the Python package `civitai_search` (indexing, retrieval methods A–F, the RAG prompt
assistant, the Gradio app and all evaluation scripts), the evaluation query sets and labels (`eval/`), and the
metric outputs (`artifacts/eval/`).

See the [top-level README](../README.md) for the architecture, results, how to run and limitations.

Quick start (from this folder):

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
export CIVITAI_DATA_DIR=..          # folder with meta.jsonl + images/ (built by ../collect.py)
PYTHONPATH=. python -m civitai_search.app --port 7860
```
