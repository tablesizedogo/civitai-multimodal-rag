---
title: Civitai Prompt Search + RAG Assistant
emoji: 🔎
colorFrom: indigo
colorTo: pink
sdk: gradio
app_file: app.py
pinned: false
license: mit
short_description: Multimodal search over 98k SFW Civitai images + prompt assistant
---

Text (EN/RU) or image → similar AI-generated images from Civitai with their real prompts and settings,
plus a prompt assistant (retrieval + local Qwen2.5-1.5B-Instruct) that cites the images it used.

Only embeddings and metadata are hosted here. Images are loaded from the Civitai CDN and belong to their
authors; every result links to its Civitai page. Code: see the GitHub repository.
