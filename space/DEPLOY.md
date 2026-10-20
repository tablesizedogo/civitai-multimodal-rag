# Deploying the demo to Hugging Face Spaces (plan, not executed)

Nothing in this folder has been uploaded. Steps when you decide to publish:

1. Build the bundle (about 280 MB: 3 embedding matrices in fp16 + slim metadata):
   ```bash
   cd search && PYTHONPATH=. .venv/bin/python ../space/prepare_space.py --out ../space/build
   ```
2. Create a Space (SDK: Gradio, hardware: CPU basic, free) at https://huggingface.co/new-space.
3. Push `space/build/` to it. The `.npy` files and `meta.jsonl` are larger than 10 MB, so enable Git LFS
   (`git lfs track "*.npy" "*.jsonl"`) or upload with `huggingface-cli upload <user>/<space> space/build .`.
   Alternative: put `data/` and `artifacts/` into a separate HF *dataset* repo and download them at startup
   with `huggingface_hub.snapshot_download` (keeps the Space repo small).
4. First start downloads CLIP (~600 MB), mCLIP (~1.1 GB), e5-small (~0.5 GB) and, on the first assistant
   request, the GGUF LLM (~1.1 GB). Cold start ≈ 2–4 min; free Spaces sleep after ~48 h without traffic.
5. RAM budget on the free tier (16 GB): ~4.5 GB for the three retrievers + ~1.5 GB for the LLM.
   With `--no-e5` (methods B/D/F removed) it drops to ~3.5 GB.

Licensing: images are not re-hosted; the gallery hotlinks `image.civitai.com` thumbnails (width=512) and each
result links to its Civitai page. Prompts are short metadata shown for search/attribution. If an author asks for
removal, delete the id from `data/meta.jsonl` and rebuild.
